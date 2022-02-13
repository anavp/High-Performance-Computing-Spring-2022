#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#define UPPER_LIMIT_N 1000
#define ITER_LIMIT 5000
#define MAX_RES_DECREASE 1000000

typedef enum method{ JACOBI, GAUSS_SEIDEL} method;

typedef long long int ll;

int getNum(char ch){
    return ((int)(ch - '0'));
}


void printVector(double vec[], int n){
    for (int i = 0; i < n; ++i)
        printf("%lf ", vec[i]);
    printf("\n");
}


int readInt(char ar[]){
    int n = 0, i = 0;
    while(ar[i] != '\0')
        n = n*10 + getNum(ar[i++]);
    return n;
}


void init(double A[][UPPER_LIMIT_N], double f[], double u[], int n){
    for (int i = 0; i < n; ++i){
        f[i] = 1;
        // TODO: 0 init?
        u[i] = 0;
        for (int j = 0; j < n; ++j){
            if (i == j) A[i][j] = 2;
            else if (abs(i - j) == 1) A[i][j] = -1;
            else A[i][j] = 0;
        }
    }
}


void deepCopyMatrix(double src[][UPPER_LIMIT_N], double dest[][UPPER_LIMIT_N], int n){
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            dest[i][j] = src[i][j];
}


void deepCopyVector(double src[], double dest[], int n){
    for (int i = 0; i < n; ++i)
            dest[i] = src[i];
}


double compute2Norm(double vec[], int n){
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += vec[i]*vec[i];
    return sqrt(sum);
}


void multiply(double matrix[][UPPER_LIMIT_N], double vec[], double ans[], int n){
    for (int i = 0; i < n; ++i){
        ans[i] = 0;
        for (int j = 0; j < n; ++j){
            ans[i] += matrix[i][j] * vec[j];
        }
    }
}


// stores the answer in vec1
void subtract(double vec1[], double vec2[], int n){
    for (int i = 0; i < n; ++i)
        vec1[i] = vec1[i] - vec2[i];
}


void subtractAndStore(double vec1[], double vec2[], double ans[], int n){
    for (int i = 0; i < n; ++i)
        ans[i] = vec1[i] - vec2[i];
}


double computeResidual(double A[][UPPER_LIMIT_N], double f[], double u[], int n){
    double ans[n];
    multiply(A, u, ans, n);
    subtract(ans, f, n);
    return compute2Norm(ans, n);
}


bool checkIfDone(double initialResidual, double currentResidual, int iterCount){
    if (iterCount >= ITER_LIMIT) return true;
    if (initialResidual - currentResidual >= MAX_RES_DECREASE) return true;
    return false;
}


void jacobiUpdate(double A[][UPPER_LIMIT_N], double f[], double u[], int n){
    double prevU[n];
    deepCopyVector(u, prevU, n);
    for (int i = 0; i < n; ++i){
        u[i] = 0;
        for (int j = 0; j < n; ++j){
            if (j == i) continue;
            u[i] += (A[i][j] * prevU[j]);
        }
        u[i] = (f[i] - u[i])/A[i][i];
    }
}


void gaussSeidelUpdate(double A[][UPPER_LIMIT_N], double f[], double u[], int n){
    for (int i = 0; i < n; ++i){
        u[i] = 0;
        for (int j = 0; j < n; ++j){
            if (j == i) continue;
            u[i] += (A[i][j] * u[j]);
        }
        u[i] = (f[i] - u[i])/A[i][i];
    }
}


void update(double A[][UPPER_LIMIT_N], double f[], double u[], int n, method updateMethod){
    switch(updateMethod){
        case JACOBI: jacobiUpdate(A, f, u, n); break;
        case GAUSS_SEIDEL: gaussSeidelUpdate(A, f, u, n); break;
    }
}


void solve(double A[][UPPER_LIMIT_N], double f[], double u[], int n){
    double initialResidual = computeResidual(A, f, u, n), currentResidual;
    currentResidual = initialResidual;
    int iterCount = 0;
    while(!checkIfDone(initialResidual, currentResidual, iterCount)){
        iterCount++;
        update(A, f, u, n, JACOBI);
        currentResidual = computeResidual(A, f, u, n);
    }
}


int main(int argc, char *argv[]){
    assert(argc != 1);
    int n = readInt(argv[1]);
    double A[n][UPPER_LIMIT_N], f[n], u[n];
    init(A, f, u, n);
    solve(A, f, u, n);
    printVector(u, n);
    return 0;
}