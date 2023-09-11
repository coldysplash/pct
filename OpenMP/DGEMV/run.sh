#!/usr/bin/bash

for ((n=15000; n <= 40000; n = $[ $n+5000 ]))
do
./dgemv-serial 1 $n
for ((p=2; p <= 8; p = $[ $p+2 ]))
do
./dgemv-parallel $p $n
done
done
