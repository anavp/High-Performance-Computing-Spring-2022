#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#define ITER_LIMIT 100
#define MAX_RES_DECREASE 1000000.0
typedef enum method{JACOBI, GAUSS_SEIDEL} method;

int getNum(char ch){ return ((int)(ch - '0')); }

int readInt(char ar[]){
    int n = 0, i = 0;
    while(ar[i] != '\0') n = n*10 + getNum(ar[i++]);
    return n;
}

int getIndex(int i, int j, int n){ return (i * n + j); }

void init(double *A, double *f, double *u, int n){
    for (int i = 0; i < n; ++i){
        f[i] = 1;
        u[i] = 0;
        for (int j = 0; j < n; ++j){
            if (i == j) A[getIndex(i, j, n)] = 2;
            else if (abs(i - j) == 1) A[getIndex(i, j, n)] = -1;
            else A[getIndex(i, j, n)] = 0;
        }
    }
}

void deepCopyVector(double *src, double *dest, int n){
    for (int i = 0; i < n; ++i)
            dest[i] = src[i];
}

double computeResidual(double *A, double *f, double *u, int n){
    double norm = 0, temp;
    for (int i = 0; i < n; ++i){
        temp = A[getIndex(i, i, n)] * u[i];
        if (i > 0) temp += A[getIndex(i, i - 1, n)] * u[i-1];
        if (i < n - 1) temp += A[getIndex(i, i + 1, n)] * u[i+1];
        temp -= f[i];
        norm += temp * temp;
    }
    return sqrt(norm);
}

bool checkIfDone(double initialResidual, double currentResidual, int iterCount){
    if (iterCount >= ITER_LIMIT) return true;
    return (currentResidual <= initialResidual/((double)MAX_RES_DECREASE));
}

void update(double *A, double *f, double *u, int n, method updateMethod){
    double *prevU;
    if (updateMethod == JACOBI){
        prevU = malloc(n * sizeof(double));
        deepCopyVector(u, prevU, n);
    }
    for (int i = 0; i < n; ++i){
        u[i] = 0;
        if (i > 0){
            if (updateMethod == JACOBI)
                u[i] += A[(getIndex(i, i-1, n))] * prevU[i-1];
            else
                u[i] += A[(getIndex(i, i-1, n))] * u[i-1];
        }
        if (i < n - 1) u[i] += A[getIndex(i, i + 1, n)] * u[i+1];
        u[i] = (f[i] - u[i])/A[getIndex(i, i, n)];
    }
    if (updateMethod == JACOBI) free(prevU);
}

void solve(double *A, double *f, double *u, int n, method updateMethod){
    double initialResidual = computeResidual(A, f, u, n), currentResidual;
    currentResidual = initialResidual;
    int iterCount = 0;
    printf("iteration = 0, initial residual = %lf\n", currentResidual);
    while(!checkIfDone(initialResidual, currentResidual, iterCount)){
        iterCount++;
        update(A, f, u, n, updateMethod);
        currentResidual = computeResidual(A, f, u, n);
        printf("iteration = %d, residual = %lf\n", iterCount, currentResidual);
    }
    if (iterCount != ITER_LIMIT)
        printf("Residual decreased by a factor of 1000000 in %d iterations\n", iterCount);
}

method getUpdateMethod(int val){
    switch(val){
        case 0: return JACOBI;
        case 1: return GAUSS_SEIDEL;
    }
    return JACOBI;
}

int main(int argc, char *argv[]){
    int n = readInt(argv[1]);
    method updateMethod = getUpdateMethod(((argc == 2)? readInt(argv[2]) : 0));
    double *A, *f, *u;
    A = malloc((n*n) * sizeof(double));
    u = malloc((n) * sizeof(double));
    f = malloc((n) * sizeof(double));
    init(A, f, u, n);
    clock_t start = clock();
    solve(A, f, u, n, updateMethod);
    clock_t end = clock();
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time taken: %f\n", cpu_time_used);
    free(A); free(u); free(f);
    return 0;
}