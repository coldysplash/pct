#!/usr/bin/bash

gcc -D_RUN_SERIAL -Wall -fopenmp main.c -o dgemv-serial
gcc -D_RUN_PARRALEL -Wall -fopenmp main.c -o dgemv-parallel
