/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN];

// Fixed bug by making sum a global variable so that it can be shared in the dotprod function.
float sum;

// We also need to make the return type here void, because
// we don't return anything.
void dotprod ()
{
int i,tid;
// Commenting this line since sum has been made a global variable
// float sum;

tid = omp_get_thread_num();
#pragma omp for reduction(+:sum)
  for (i=0; i < VECLEN; i++)
    {
    sum = sum + (a[i]*b[i]);
    printf("  tid= %d i=%d\n",tid,i);
    }
}


int main (int argc, char *argv[]) {
int i;
// Commenting this line since sum is a global variable now
// float sum;

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;
// sum = 0.0;

#pragma omp parallel shared(sum)
  dotprod();

printf("Sum = %f\n",sum);

}
