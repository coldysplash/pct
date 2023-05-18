#!/bin/bash

for ((p=1; p <= 6; p = $[ $p+1 ]))
do
echo "p = $p"
./mc.out $p
done
