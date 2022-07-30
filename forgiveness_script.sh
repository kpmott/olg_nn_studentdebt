#!/bin/bash

for g in 0 1 2 3 4 5 6 7
do
echo "$g"
./rbar_parallel.py -G $g
done