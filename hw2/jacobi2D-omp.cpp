
#include <stdio.h>
#include <cmath>
#ifdef _OPENMP
  #include <omp.h>
#endif
#include "utils.h"


void jacobi_serial(double* u, double* f, long N, int maxiter){
  double hh = 1.0/(double)((N+1)*(N+1)), norm, threshold;
  long N2 = N+2;
  double *u_prev = (double*) malloc(N2*N2 * sizeof(double));

  for (int i=0; i<maxiter; i++) {
    for (long j = 0; j < N2*N2; j++) 
      u_prev[j] = u[j];
    norm = 0;

    for (long k=1; k<N+1; k++)
      for (long l=1; l<N+1; l++) {
        u[k * N2+l] = 0.25 * (hh * f[k * N2+l] + u_prev[(k-1) * N2 + l] + u_prev[k * N2+(l-1)] + u_prev[(k+1) * N2+l] + u_prev[k * N2+(l+1)]);
        norm += (u[k*N2+l] - u_prev[k * N2+l]) * (u[k * N2+l] - u_prev[k * N2+l]);
      }
    norm = sqrt(norm);
    if (i == 0)
      threshold = 1e-6 * norm;
    if (norm < threshold) {
      printf("Threshold reached after %d iterations! norm=%e\n", i, norm);
      break;
    }
    if (i==maxiter-1) 
      printf("Maxiter reached (%d iterations)! norm=%e\n", maxiter, norm);
  }

  free(u_prev);
}


void jacobi_parallel(double* u, double* f, long N, int maxiter){
  double hh, norm, threshold;
  const long N2 = N+2;
  double* u_prev = (double*) malloc(N2*N2 * sizeof(double));
  hh = 1.0/(double)((N+1)*(N+1));
  #ifdef _OPENMP
  printf("Num threads: %d\n", omp_get_max_threads());
  #endif
  for (int k=0; k<maxiter; k++) {
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

    if (k==0){
      threshold = 1e-6 * norm;
    }
    if (norm < threshold) {
      printf("Threshold reached after %d iterations! norm=%e\n", k, norm);
      break;
    }
    if (k==maxiter-1) printf("Max iter count reached (%d iterations). norm=%e\n", maxiter, norm);
  }

  free(u_prev);
}


int main(int argc, char** argv) {
  
  const long N = 100;
  const long N2 = N+2;
  int maxiter = 1000;

  double* f = (double*) malloc(N2*N2 * sizeof(double));
  for (long ij = 0; ij < N2*N2; ij++) f[ij] = 1;

  double* u = (double*) malloc(N2*N2 * sizeof(double));
  double* u_par = (double*) malloc(N2*N2 * sizeof(double));
  for (long ij = 0; ij < N2*N2; ij++) u[ij] = 0;
  for (long ij = 0; ij < N2*N2; ij++) u_par[ij] = 0;

  double tt, time;
  Timer timer;
  #ifdef _OPENMP
  tt = omp_get_wtime();
  #else
  timer.tic();
  #endif
  jacobi_serial(u, f, N, maxiter);
  #ifdef _OPENMP
  time = omp_get_wtime() - tt;
  #else
  time = timer.toc();
  #endif
  printf("jacobi serial = %fs\n", time);
  
  #ifdef _OPENMP
  tt = omp_get_wtime();
  #else
  timer.tic();
  #endif
  jacobi_parallel(u_par, f, N, maxiter);
  #ifdef _OPENMP
  time = omp_get_wtime() - tt;
  #else
  time = timer.toc();
  #endif
  printf("jacobi parallel = %fs\n", time);
   
  free(u);
  free(u_par);
  free(f);

  return 0;
}