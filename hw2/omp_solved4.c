/******************************************************************************
* FILE: omp_bug4.c
* DESCRIPTION:
*   This very simple program causes a segmentation fault.
* AUTHOR: Blaise Barney  01/09/04
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1048

int main (int argc, char *argv[]) 
{
int nthreads, tid, i, j;
// Fixed bug by making a a pointer because a 2d array defined from the stack leads to out of memory in stack error because each thread creates a copy of the same. That is avoided if we dynamically allocated memory for a from the heap.
//double a[N][N];
double *a;

/* Fork a team of threads with explicit variable scoping */
#pragma omp parallel shared(nthreads) private(i,j,tid,a)
  {
  // Dynamically allocating memory.
  a = (double *) malloc(N*N*sizeof(double)); 
  /* Obtain/print thread info */
  tid = omp_get_thread_num();
  if (tid == 0) 
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d starting...\n", tid);

  /* Each thread works on its own private copy of the array */
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      a[i*N + j] = tid + i + j;

  /* For confirmation */
  printf("Thread %d done. Last element= %lf\n",tid,a[(N-1)*N + N-1]);
  // Freeing the dynamically allocated memory of a ptr
  free(a);
  }  /* All threads join master thread and disband */
  return 0;
}

