#include <bits/stdc++.h>
#include <omp.h>

#define BLOCK_SIZE 1024
#define endl "\n"
using namespace std;
typedef long long int ll;

void jacobiCPU(double* u, double* f, long N, int maxiter){
    double hh, norm, threshold;
    const long N2 = N+2;
    double* u_prev = (double*) malloc(N2*N2 * sizeof(double));
    hh = 1.0/(double)((N+1)*(N+1));

    for (int k = 0; k < maxiter; k++) {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) shared(u, u_prev)
        #endif
        for (long ij = 0; ij < N2*N2; ij++) u_prev[ij] = u[ij];
        norm = 0;

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(+:norm) shared(u, u_prev)
        #endif
        for (long i=1; i<N+1; i++) {
            for (long j=1; j<N+1; j++) {
                u[i*N2+j] = 0.25*(hh*f[i * N2+j] + u_prev[(i-1) * N2 + j] + u_prev[i * N2+(j-1)] + u_prev[(i+1) * N2+j] + u_prev[i * N2+(j+1)]);
                norm += (u[i * N2+j] - u_prev[i * N2+j]) * (u[i * N2+j] - u_prev[i * N2+j]);
            }
        }
        norm = sqrt(norm);
        if (k==0)
            threshold = 1e-6 * norm;
        if (norm < threshold) {
            printf("Threshold reached after %d iterations! norm=%e\n", k, norm);
            break;
        }
        if (k==maxiter-1) printf("Max iter count reached (%d iterations). norm=%e\n", maxiter, norm);
    }
    free(u_prev);
}

__global__ void reduction_v2(double* sum, const double* a, long N) {
    __shared__ double sharedMemory[BLOCK_SIZE];
    int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < N) sharedMemory[threadIdx.x] = a[idx];
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
        if (threadIdx.x == 0) sum[blockIdx.x] = sharedMemory[0] + sharedMemory[1];
    }
}

__global__ void update(double* u, double* u_temp, const double* f, int N) {
    double h = 1 / (N + 1.0);
    int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
    int j = idx / (N + 2);
    int i = idx % N + 2;
    if (0 < j && j < N + 1 && 0 < i && i < N + 1) {
        u_temp[idx] = h*h*f[idx] + u[idx - 1] + u[idx + 1] + u[idx - (N + 2)] + u[idx + (N + 2)];
        u_temp[idx] = 0.25*u_temp[idx];
    }
}

double redux(double* a, long N) {
    double *y_d;
    long N_work = 1;
    for (long i = (N + BLOCK_SIZE - 1) / (BLOCK_SIZE); i > 1; i = (i + BLOCK_SIZE - 1) / (BLOCK_SIZE)) N_work += i;
    cudaMalloc(&y_d, N_work * sizeof(double));

    double* sum_d = y_d;
    long N_rem = (N + BLOCK_SIZE - 1) / (BLOCK_SIZE);
    reduction_v2 << <N_rem, BLOCK_SIZE >> >(sum_d, a, N);
    while (N_rem > 1) {
        long N = N_rem;
        N_rem = (N_rem + BLOCK_SIZE - 1) / (BLOCK_SIZE);
        reduction_v2 << <N_rem, BLOCK_SIZE >> >(sum_d + N, sum_d, N);
        sum_d += N;
    }

    double sum;
    cudaMemcpyAsync(&sum, sum_d, 1 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return sum;
}

void jacobiGPU(int N, int maxIters){
    int N2 = N + 2;
    double *u, *f;
    cudaMallocHost((void**)&u, N2 * N2 * sizeof(double));
    cudaMallocHost((void**)&f, N2 * N2 * sizeof(double)); 
    for (long i = 0; i < N2 * N2; ++i) {
        f[i] = 1.0;
        u[i] = 0.0;
    }

    double *deviceF, *deviceResidualArray;
    double *deviceU, *deviceTempU;

    cudaMalloc(&deviceF, N2 * N2 * sizeof(double));
    cudaMalloc(&deviceResidualArray, N2 * N2 * sizeof(double));
    cudaMalloc(&deviceU, N2 * N2 * sizeof(double));
    cudaMalloc(&deviceTempU, N2 * N2 * sizeof(double));
    
    double time = omp_get_wtime();
    cudaMemcpyAsync(deviceF, f, N2 * N2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(deviceU, u, N2 * N2 * sizeof(double), cudaMemcpyHostToDevice);    
    cudaDeviceSynchronize();
    
    long N_rem = N2 * N2 / BLOCK_SIZE;

    for (int i = 0; i < maxIters; i++) {
        update <<<N_rem, BLOCK_SIZE>>>(deviceU, deviceTempU, f, N);
        cudaDeviceSynchronize();
        double *uTemp = deviceTempU;
        deviceTempU = deviceU;
        deviceU = uTemp;
    }

    cudaFree(deviceU);
    cudaFree(deviceTempU);
    cudaFree(deviceResidualArray);
    cudaFree(deviceF); 
    cudaFreeHost(u);
    cudaFreeHost(f);
}

void Check_CUDA_Error(const char *message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

int main() {
    int numThreads;
    cout << "Input number of threads: ";
    cin >> numThreads;
    cudaDeviceProp dev;
    cudaGetDeviceProperties(&dev, 0);
    cout << "Device Name: " << dev.name << endl;
    cout << "Computation Capability: " << dev.major << "." << dev.minor << endl;
    cout << "Memory: " << dev.totalGlobalMem * 1.0e-9 << endl;
    cout << endl;
    ll N = 1e3, maxIters = 21;
    cout << "Input N: ";
    cin >> N;
    cout << "Input maxIters: ";
    cin >> maxIters;
    ll N2 = N+2;
    double *u, *f;
    u = (double*) malloc(N2 * N2 * sizeof(double));
    f = (double*) malloc(N2 * N2 * sizeof(double));

    // Initialization:
    memset(u, 0, N2 * N2 * sizeof(double));
    memset(f, 0, N2 * N2 * sizeof(double));
    for (int i = 0; i < N2; ++i)
        f[i] = 1.0;
    double time = omp_get_wtime();
    jacobiCPU(u, f, N, maxIters);
    double elapsed_time = (omp_get_wtime() - time);
    cout << "CPU Bandwidth = " << maxIters * 10 * N2 * N2 * sizeof(double) / (elapsed_time) / 1e9 << " GB/s" << endl;
    cout << "CPU Total Time = " << elapsed_time << "s" << endl;
    cout << endl;

    memset(u, 0, N2 * N2 * sizeof(double));
    time = omp_get_wtime();
    jacobiGPU(N, maxIters);
    elapsed_time = (omp_get_wtime() - time);
    cout << "GPU Bandwidth = " << maxIters * 10 * N2 * N2 * sizeof(double) / (elapsed_time) / 1e9 << " GB/s" << endl;
    cout << "GPU Total Time = " << elapsed_time << "s" << endl;

    free(u);
    free(f);
    return 0; 
}