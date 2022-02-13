#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include<unistd.h>
#define ITER_LIMIT 5000
#define MAX_RES_DECREASE 1000000

typedef enum method{JACOBI, GAUSS_SEIDEL} method;


int getNum(char ch){
    return ((int)(ch - '0'));
}


void printVector(double *vec, int n){
    for (int i = 0; i < n; ++i)
        printf("%lf ", vec[i]);
    printf("\n");
}


int readInt(char ar[]){
    int n = 0, i = 0;
    while(ar[i] != '\0') n = n*10 + getNum(ar[i++]);
    return n;
}


int getIndex(int i, int j, int n){
    return (i * n + j);
}


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


double compute2Norm(double *vec, int n){
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += vec[i]*vec[i];
    return sqrt(sum);
}


void multiply(double *matrix, double *vec, double *ans, int n){
    for (int i = 0; i < n; ++i){
        ans[i] = 0;
        for (int j = 0; j < n; ++j)
            ans[i] += matrix[getIndex(i, j, n)] * vec[j];
    }
}


void subtract(double *vec1, double *vec2, int n){
    for (int i = 0; i < n; ++i)
        vec1[i] = vec1[i] - vec2[i];
}


double computeResidual(double *A, double *f, double *u, int n){
    double ans[n];
    multiply(A, u, ans, n);
    subtract(ans, f, n);
    return compute2Norm(ans, n);
}


bool checkIfDone(double initialResidual, double currentResidual, int iterCount){
    if (iterCount >= ITER_LIMIT) return true;
    return (initialResidual - currentResidual >= MAX_RES_DECREASE);
}


void update(double *A, double *f, double *u, int n, method updateMethod){
    double *prevU;
    if (updateMethod == JACOBI){
        prevU = malloc(n * sizeof(double));
        deepCopyVector(u, prevU, n);
    }
    for (int i = 0; i < n; ++i){
        u[i] = 0;
        for (int j = 0; j < n; ++j){
            if (j == i) continue;
            if (updateMethod == JACOBI)
                u[i] += (A[getIndex(i, j, n)] * prevU[j]);
            else
                u[i] += (A[getIndex(i, j, n)] * u[j]);
        }
        u[i] = (f[i] - u[i])/A[getIndex(i, i, n)];
    }
}


void solve(double *A, double *f, double *u, int n, method updateMethod){
    double initialResidual = computeResidual(A, f, u, n), currentResidual;
    currentResidual = initialResidual;
    int iterCount = 0;
    printf("initial residual = %lf\n", currentResidual);
    while(!checkIfDone(initialResidual, currentResidual, iterCount)){
        iterCount++;
        update(A, f, u, n, updateMethod);
        currentResidual = computeResidual(A, f, u, n);
        printf("residual = %lf\n", currentResidual);
    }
}


method getUpdateMethod(int val){
    switch(val){
        case 0: return JACOBI;
        case 1: return GAUSS_SEIDEL;
    }
    return JACOBI;
}


int main(int argc, char *argv[]){
    assert(argc == 3);
    int n = readInt(argv[1]);
    method updateMethod = getUpdateMethod(readInt(argv[2]));
    double *A, *f, *u;
    A = malloc((n*n) * sizeof(double));
    u = malloc((n) * sizeof(double));
    f = malloc((n) * sizeof(double));
    init(A, f, u, n);
    solve(A, f, u, n, updateMethod);
    return 0;
}