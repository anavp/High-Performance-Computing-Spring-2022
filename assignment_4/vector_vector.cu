#include <bits/stdc++.h>
#include <omp.h>

#define BLOCK_SIZE 1024
#define endl "\n"
using namespace std;
typedef long long int ll;


double cpuProduct(const double* a, const double* b, ll N){
    double sum = 0;
    #pragma omp parallel for schedule(static) reduction(+:sum)
    for (ll i = 0; i < N; i++) 
        sum += a[i]*b[i];
    return sum;
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
    ll N = (1UL<<24);

    double *vector1, *vector2;
    cudaMallocHost((void**)&vector1, N * sizeof(double));
    cudaMallocHost((void**)&vector2, N * sizeof(double));
    
    // Initialization:
    #pragma omp parallel for schedule(static)
    for (ll i = 0; i < N; i++) {
        vector1[i] = ((double)rand())/((double)rand());
        vector2[i] = ((double)rand())/((double)rand());
    }

    double referenceSum, sum;
    double time = omp_get_wtime();
    // CPU Computation
    referenceSum = cpuProduct(vector1, vector2, N);
    cout << "CPU Bandwidth = " << N * sizeof(double) / (omp_get_wtime() - time) / 1e9 << " GB/s" << endl;

    // GPU Computation starts here:
    double *deviceVector1, *deviceVector2, *intermediateVector;
    // Allocate memory on device
    cudaMalloc(&deviceVector1, N*sizeof(double));
    cudaMalloc(&deviceVector2, N*sizeof(double));
    ll N_work = 1;
    for (ll i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) 
        N_work += i;
    cudaMalloc(&intermediateVector, N_work * sizeof(double)); // extra memory buffer for reduction across thread-blocks

    // Copy to values to device
    cudaMemcpyAsync(deviceVector1, vector1, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(deviceVector2, vector2, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    time = omp_get_wtime();

    double* sum_d = intermediateVector;
    ll N_rem = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
    // Multiply and Reduce!
    reduction_product<<<N_rem,BLOCK_SIZE>>>(sum_d, deviceVector1, deviceVector2, N, false);
    
    // Reduce
    while (N_rem > 1) {
        ll N = N_rem;
        N_rem = (N_rem+BLOCK_SIZE-1)/(BLOCK_SIZE);
        reduction_product<<<N_rem,BLOCK_SIZE>>>(sum_d + N, sum_d, NULL, N);
        sum_d += N;
    }

    cudaMemcpyAsync(&sum, sum_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Outputs:
    cout << "GPU Bandwidth = " << N * sizeof(double) / (omp_get_wtime() - time) / 1e9 << endl;
    cout << fixed;
    cout << setprecision(4) << "Error = " << abs(sum - referenceSum) << endl;

    // Free allocated memory
    cudaFree(deviceVector1);
    cudaFree(deviceVector2);
    cudaFreeHost(vector1);
    cudaFreeHost(vector2);

    return 0;
}