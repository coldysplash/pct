#!/bin/bash

for ((p=2; p <= 8; p = $[ $p+2 ]))
do
echo "p = $p"
./midpoint $p
done
