#include <bits/stdc++.h>
#include <omp.h>

#define BLOCK_SIZE 1024
#define endl "\n"
using namespace std;
typedef long long int ll;

inline ll getIndex(ll i, ll j, ll N){
    return N * i + j;
}

void cpuMatrixVectorMult(double* ans, const double* mat, const double* vec, ll N, ll M) {
    for (ll i = 0; i < M; i++){
        double sum = 0;
        #pragma omp parallel for schedule(static) reduction(+:sum)
        for (ll j = 0; j < N; j++)
            sum += mat[i * N + j] * vec[j];
        ans[i] = sum;
    }
}

__global__ void gpuMatrixVectorMult(double* intermediateMat, const double* mat, const double* vec, ll N, ll M) {
    ll idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    ll idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_x < N && idx_y < M)
        intermediateMat[idx_x * M + idx_y] = mat[idx_y * N + idx_x] * vec[idx_x];
}


__global__ void reduction_v2(double* sum, const double* a, ll N, ll M) {
    __shared__ double sharedMemory[BLOCK_SIZE];
    ll idx = (blockIdx.x) * blockDim.x + threadIdx.x;
    ll idx_y = blockIdx.y;
    if (idx < N) sharedMemory[threadIdx.x] = a[M * idx + idx_y];
    else sharedMemory[threadIdx.x] = 0;

    __syncthreads();
    if (threadIdx.x < 512) sharedMemory[threadIdx.x] += sharedMemory[threadIdx.x + 512];
    __syncthreads();
    if (threadIdx.x < 256) sharedMemory[threadIdx.x] += sharedMemory[threadIdx.x + 256];
    __syncthreads();
    if (threadIdx.x < 128) sharedMemory[threadIdx.x] += sharedMemory[threadIdx.x + 128];
    __syncthreads();
    if (threadIdx.x <  64) sharedMemory[threadIdx.x] += sharedMemory[threadIdx.x + 64];
    __syncthreads();
    if (threadIdx.x <  32) {
        sharedMemory[threadIdx.x] += sharedMemory[threadIdx.x + 32];
        __syncwarp();
        sharedMemory[threadIdx.x] += sharedMemory[threadIdx.x + 16];
        __syncwarp();
        sharedMemory[threadIdx.x] += sharedMemory[threadIdx.x + 8];
        __syncwarp();
        sharedMemory[threadIdx.x] += sharedMemory[threadIdx.x + 4];
        __syncwarp();
        sharedMemory[threadIdx.x] += sharedMemory[threadIdx.x + 2];
        __syncwarp();
        if (threadIdx.x == 0) {
            sum[M * blockIdx.x + idx_y] = sharedMemory[0] + sharedMemory[1];
        }
    }
}

int main() {
    srand(rand());
    cudaDeviceProp dev;
    cudaGetDeviceProperties(&dev, 0);
    cout << "Device Name: " << dev.name << endl;
    cout << "Computation Capability: " << dev.major << "." << dev.minor << endl;
    cout << "Memory: " << dev.totalGlobalMem * 1.0e-9 << endl;
    cout << "Peak Memory Bandwidth (GB/s): " << 2.0 * dev.memoryClockRate * (dev.memoryBusWidth/8)/1.0e6;
    cout << endl << endl;
    
    ll N = 1e5, M = 2e2, blockSize = 32;
    ll gridDimsX = (N + blockSize - 1) / blockSize, gridDimsY = (M + blockSize - 1) / blockSize;
    dim3 blockDims(blockSize, blockSize), gridDims(gridDimsX, gridDimsY);
    
    double *mat, *vec, *ans, *referenceAns;
    referenceAns = (double* ) malloc(M*sizeof(double));
    cudaMallocHost((void**)&mat, N * M * sizeof(double));
    cudaMallocHost((void**)&vec, N * sizeof(double));
    cudaMallocHost((void**)&ans, M * sizeof(double));

    // Initialization:
    for (ll i = 0; i < N; ++i){
        vec[i] = ((double)rand())/((double)rand());
        for (ll j = 0; j < M; ++j)
            mat[getIndex(j, i, N)] = ((double)rand())/((double)rand());
    }
    for (ll i = 0; i < M; ++i){
        ans[i] = 0;
        referenceAns[i] = 0;
    }

    double time = omp_get_wtime();    
    // CPU Computation
    cpuMatrixVectorMult(referenceAns, mat, vec, N, M);
    cout << "CPU Bandwidth = " << (M * N * sizeof(double)) / (omp_get_wtime() - time)/1e9 << " GB/s" << endl;
    
    // GPU Computation starts here:
    double *deviceMat, *deviceVec, *deviceAns, *intermediateMat; 
    // Allocate memory on device
    cudaMalloc(&deviceMat, N * M * sizeof(double));
    cudaMalloc(&intermediateMat, N * M * sizeof(double));
    cudaMalloc(&deviceVec, N * sizeof(double));
    time = omp_get_wtime();
    // Copy to values to device
    cudaMemcpyAsync(deviceMat, mat, N * M * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(deviceVec, vec, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Multiply!
    gpuMatrixVectorMult <<<gridDims, blockDims>>> (intermediateMat, deviceMat, deviceVec, N, M);
    cudaDeviceSynchronize();

    ll N_work = M;
    for (ll i = (N + BLOCK_SIZE - 1) / (BLOCK_SIZE); i > 1; i = (i + BLOCK_SIZE - 1) / (BLOCK_SIZE)) 
        N_work += i*M;

    double *tempVec;
    cudaMalloc(&tempVec, N_work * sizeof(double));
    deviceAns = tempVec;
    ll N_rem = (N + BLOCK_SIZE - 1) / (BLOCK_SIZE);
    dim3 gridDims2(N_rem, M), blockDims2(BLOCK_SIZE, 1);
    // Reduce
    reduction_v2 <<<gridDims2, blockDims2>>> (deviceAns, intermediateMat, N, M);
    cudaDeviceSynchronize();
    
    // Reduce
    while (N_rem > 1) {
        ll N = N_rem;
        N_rem = (N_rem + BLOCK_SIZE - 1) / (BLOCK_SIZE);
        dim3 gridDims(N_rem, M);
        reduction_v2 <<<gridDims, blockDims2>>> (deviceAns + N*M, deviceAns, N, M);
        cudaDeviceSynchronize();
        deviceAns += N * M;
    }
    cudaDeviceSynchronize();
    
    // Get the answer
    cudaMemcpyAsync(ans, deviceAns, M * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Outputs:
    cout << "GPU Bandwidth = " << N * M * sizeof(double) / (omp_get_wtime() - time) / 1e9 << " GB/s" << endl;
    double error = 0.0;    
    for (ll i = 0; i < M; ++i)
        error += (ans[i] - referenceAns[i]) * (ans[i] - referenceAns[i]);
    cout << fixed;
    cout << setprecision(4) << "Error = " << error << endl;
    
    // Free allocated memory
    cudaFreeHost(mat);
    cudaFreeHost(vec);
    cudaFreeHost(ans);
    free(referenceAns);
    cudaFree(deviceMat);
    cudaFree(deviceVec);
    cudaFree(tempVec);
    cudaFree(intermediateMat);
    return 0; 
}