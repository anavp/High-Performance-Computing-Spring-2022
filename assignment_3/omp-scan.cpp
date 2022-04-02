#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define p 10
// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}


void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  int nthreads, block_len;
  int num_threads, tid;
  const int block_size = n/p;
  int private_ind;
  #pragma omp parallel num_threads(p) private(tid,private_ind) shared(prefix_sum, A)
  {
    tid = omp_get_thread_num();
    private_ind = tid * block_size;
    prefix_sum[private_ind++] = 0;
    // #pragma omp parallel for
    for (; private_ind < (tid+1)*block_size; ++private_ind)
      prefix_sum[private_ind] = prefix_sum[private_ind - 1] + A[private_ind-1];
  }

  // #pragma omp parallel for
  for (long i = 1; i < p; ++i){
    for (long j = i * block_size; j < (i + 1) * block_size && j < n; ++j){
      prefix_sum[j] += prefix_sum[i*block_size - 1] + A[i*block_size - 1];
    }
  }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
