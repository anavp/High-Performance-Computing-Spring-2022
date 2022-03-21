#include <stdio.h>
#include <cmath>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include "utils.h"


void gauss_seidel(double* u, double* f, long N, int maxiter){
  double hh = 1.0/(double)((N+1)*(N+1)), norm, threshold, u_prev;
  const long N2 = N+2;

  for (int k=0; k<maxiter; k++) {
    norm = 0;
    for (long i=1; i<N+1; i++) {
      for (long j=1; j<N+1; j++) {
        u_prev = u[i*N2+j];
        u[i*N2+j] = 0.25*(hh*f[i*N2+j] + u[(i-1)*N2 + j] + u[i*N2+(j-1)] + u[(i+1)*N2+j] + u[i*N2+(j+1)]);
        norm += (u[i*N2+j] - u_prev)*(u[i*N2+j] - u_prev);
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
    if (k==maxiter-1) printf("Maxiter reached (%d iterations)! norm=%e\n", maxiter, norm);
  }

}


void gauss_seidel_colored(double* u, double* f, long N, int maxiter){
  double hh, norm, threshold, u_prev;
  long jstart, istart;
  const long N2 = N+2;
  hh = 1.0/(double)((N+1)*(N+1));

  for (int k=0; k<maxiter; k++) {
    norm = 0;
    for (long i=1; i<N+1; i++) {
      if (i%2==1) jstart = 1;
      else jstart = 2;
      for (long j=jstart; j<N+1; j+=2) {
        u_prev = u[i*N2+j];
        u[i*N2+j] = 0.25*(hh*f[i*N2+j] + u[(i-1)*N2 + j] + u[i*N2+(j-1)] + u[(i+1)*N2+j] + u[i*N2+(j+1)]);
        norm += (u[i*N2+j] - u_prev)*(u[i*N2+j] - u_prev);
      }
    }
    for (long i=1; i<N+1; i++) {
      if (i%2==1) jstart = 2;
      else jstart = 1;
      for (long j=jstart; j<N+1; j+=2) {
        u_prev = u[i*N2+j];
        u[i*N2+j] = 0.25*(hh*f[i*N2+j] + u[(i-1)*N2 + j] + u[i*N2+(j-1)] + u[(i+1)*N2+j] + u[i*N2+(j+1)]);
        norm += (u[i*N2+j] - u_prev)*(u[i*N2+j] - u_prev);
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
    if (k==maxiter-1) printf("Maxiter reached (%d iterations)! norm=%e\n", maxiter, norm);
  }

}


void gauss_seidel_colored_parallel(double* u, double* f, long N, int maxiter){
  double hh, norm, threshold, u_prev;
  long jstart, istart;
  const long N2 = N+2;
  hh = 1.0/(double)((N+1)*(N+1));
  #ifdef _OPENMP
  printf("Num threads: %d\n", omp_get_max_threads());
  #endif

  for (int k=0; k<maxiter; k++) {
    norm = 0;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(+:norm) shared(u) private(u_prev)
    #endif
    for (long i=1; i<N+1; i++) {
      if (i%2==1) jstart = 1;
      else jstart = 2;
      for (long j=jstart; j<N+1; j+=2) {
        u_prev = u[i*N2+j];
        u[i*N2+j] = 0.25*(hh*f[i*N2+j] + u[(i-1)*N2 + j] + u[i*N2+(j-1)] + u[(i+1)*N2+j] + u[i*N2+(j+1)]);
        norm += (u[i*N2+j] - u_prev)*(u[i*N2+j] - u_prev);
      }
    }
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(+:norm) shared(u) private(u_prev)
    #endif
    for (long i=1; i<N+1; i++) {
      if (i%2==1) jstart = 2;
      else jstart = 1;
      for (long j=jstart; j<N+1; j+=2) {
        u_prev = u[i*N2+j];
        u[i*N2+j] = 0.25*(hh*f[i*N2+j] + u[(i-1)*N2 + j] + u[i*N2+(j-1)] + u[(i+1)*N2+j] + u[i*N2+(j+1)]);
        norm += (u[i*N2+j] - u_prev)*(u[i*N2+j] - u_prev);
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
    if (k==maxiter-1) printf("Maxiter reached (%d iterations)! norm=%e\n", maxiter, norm);
  }

}


int main(int argc, char** argv) {
  
  const long N = 100;
  const long N2 = N+2;
  int maxiter = 1000;
  
  double* f = (double*) malloc(N2*N2 * sizeof(double));
  for (long ij = 0; ij < N2*N2; ij++) f[ij] = 1.0;

  double* u = (double*) malloc(N2*N2 * sizeof(double));
  double* u_col = (double*) malloc(N2*N2 * sizeof(double));
  double* u_par = (double*) malloc(N2*N2 * sizeof(double));
  for (long ij = 0; ij < N2*N2; ij++){
       u[ij] = 0.0;
       u_col[ij] = 0.0;
       u_par[ij] = 0.0;
  }

  double tt, time;
  Timer timer;
  #ifdef _OPENMP
  tt = omp_get_wtime();
  #else
  timer.tic();
  #endif
  gauss_seidel(u, f, N, maxiter);
  #ifdef _OPENMP
  time = omp_get_wtime() - tt;
  #else
  time = timer.toc();
  #endif
  printf("gauss-seidel (serial) = %fs\n", time);
   
  #ifdef _OPENMP
  tt = omp_get_wtime();
  #else
  timer.tic();
  #endif
  gauss_seidel_colored(u_col, f, N, maxiter);
  #ifdef _OPENMP
  time = omp_get_wtime() - tt;
  #else
  time = timer.toc();
  #endif
  printf("gauss-seidel colored (serial) = %fs\n", time);

  #ifdef _OPENMP
  tt = omp_get_wtime();
  #else
  timer.tic();
  #endif
  gauss_seidel_colored_parallel(u_par, f, N, maxiter);
  #ifdef _OPENMP
  time = omp_get_wtime() - tt;
  #else
  time = timer.toc();
  #endif
  printf("gauss-seidel colored (parallel) = %fs\n", time);
   
  free(u);
  free(u_col);
  free(u_par);
  free(f);

  return 0;
}
