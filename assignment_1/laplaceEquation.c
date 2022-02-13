#include <stdio.h>
#include <assert.h>
#define UPPER_LIMIT_N 1000

int getNum(char ch){
    return ((int)(ch - '0'));
}


int readInt(char ar[]){
    int n = 0, i = 0;
    while(ar[i] != '\0')
        n = n*10 + getNum(ar[i++]);
    return n;
}


void init(int A[][UPPER_LIMIT_N], int f[], int n){
    for (int i = 0; i < n; ++i) f[i] = 1;
}


int main(int argc, char *argv[]){
    assert(argc != 1);
    int n = readInt(argv[1]);
    int f[UPPER_LIMIT_N], A[UPPER_LIMIT_N][UPPER_LIMIT_N];
    init(A, f, n);
    return 0;
}