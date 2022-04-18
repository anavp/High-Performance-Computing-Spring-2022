#include <bits/stdc++.h>
#include <omp.h>

#define BLOCK_SIZE 1024
#define endl "\n"
using namespace std;
typedef long long int ll;

void cpuMatrixVectorMult(double* ans, const double* mat, const double* vec, ll N, ll M) {
    for (ll i = 0; i < M; i++){
        double sum = 0;
        #pragma omp parallel for schedule(static) reduction(+:sum)
        for (ll j = 0; j < N; j++)
            sum += mat[i * N + j] * vec[j];
        ans[i] = sum;
    }
}

void Check_CUDA_Error(const char *message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

__global__ void reduction_product(double* sum, const double* a, const double* b = NULL, ll N = (1UL<<24), bool isReduction = true){
    __shared__ double sharedMemory[BLOCK_SIZE];
    int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < N) sharedMemory[threadIdx.x] = (isReduction ? a[idx]: a[idx]*b[idx]);
    else sharedMemory[threadIdx.x] = 0;

    __syncthreads();
    if (threadIdx.x < 512) sharedMemory[threadIdx.x] += sharedMemory[threadIdx.x + 512];
    __syncthreads();
    if (threadIdx.x < 256) sharedMemory[threadIdx.x] += sharedMemory[threadIdx.x + 256];
    __syncthreads();
    if (threadIdx.x < 128) sharedMemory[threadIdx.x] += sharedMemory[threadIdx.x + 128];
    __syncthreads();
    if (threadIdx.x < 64) sharedMemory[threadIdx.x] += sharedMemory[threadIdx.x + 64];
    __syncthreads();
    if (threadIdx.x < 32) {
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
        if (threadIdx.x == 0) sum[blockIdx.x] = sharedMemory[0] + sharedMemory[1];
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
    ll N = (1UL<<18), M = 5e2, blockSize = 32;
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
            mat[j * N + i] = ((double)rand())/((double)rand());
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
    double *deviceMat, *deviceVec, *intermediateVec; 
    // Allocate memory on device
    cudaMalloc(&deviceMat, N * M * sizeof(double));
    cudaMalloc(&deviceVec, N * sizeof(double));
    ll N_work = 1;
    for (long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE))
        N_work += i;
    cudaMalloc(&intermediateVec, N_work * sizeof(double));
    // Copy to values to device
    cudaMemcpyAsync(deviceMat, mat, N * M * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(deviceVec, vec, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    time = omp_get_wtime();

    // Multiply
    for (long i = 0; i < M; i++) {
        double* deviceSum = intermediateVec;
        long N_rem = (N + BLOCK_SIZE - 1) / (BLOCK_SIZE);
        reduction_product<<<N_rem,BLOCK_SIZE>>>(deviceSum, deviceMat + i * N, deviceVec, N, false);
        while (N_rem > 1) {
            long tempN = N_rem;
            N_rem = (N_rem + BLOCK_SIZE - 1) / (BLOCK_SIZE);
            reduction_product<<<N_rem,BLOCK_SIZE>>>(deviceSum + tempN, deviceSum, NULL, tempN);
            deviceSum += tempN;
        }
        cudaMemcpyAsync(&ans[i], deviceSum, sizeof(double), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }
    
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
    cudaFree(intermediateVec);
    return 0; 
}