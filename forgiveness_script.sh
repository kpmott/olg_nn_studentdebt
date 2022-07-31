#!/bin/bash

for g in 0 1 2 3 4 5 6 7 8
do
echo "$g"
./forgiveness.py -G $g
done