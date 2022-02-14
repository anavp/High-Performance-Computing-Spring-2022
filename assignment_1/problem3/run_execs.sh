#!/bin/bash

./laplaceO0 100 0 > output_jacobi.log
./laplaceO3 100 1 > output_gs.log
./laplaceO0 10000 0 > output_jacobi_10000.log
./laplaceO3 10000 1 > output_gs_10000.log