#!/bin/bash

./laplaceO0 100 0 > output_jacobi_O0.log
./laplaceO3 100 0 > output_jacobi_O3.log
./laplaceO0 100 1 > output_gs_O0.log
./laplaceO3 100 1 > output_gs_O3.log
./laplaceO0 10000 0 > output_jacobi_10000_O0.log
./laplaceO3 10000 0 > output_jacobi_10000_O3.log
./laplaceO0 10000 1 > output_gs_10000_O0.log
./laplaceO3 10000 1 > output_gs_10000_O3.log
