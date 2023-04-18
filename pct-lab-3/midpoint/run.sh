#!/bin/bash

for ((p=1; p <= 5; p = $[ $p+2 ]))
do
echo "p = $p"
./midpoint.out $p
done
