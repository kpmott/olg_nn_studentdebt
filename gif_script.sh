#!/bin/bash

for g in 0 1 2 3 4 5 6 7
do
#echo "$g"
cp ./plots/$g/.e.png ./plots/gif/e$g.png
cp ./plots/$g/.b.png ./plots/gif/b$g.png
cp ./plots/$g/.exret.png ./plots/gif/exret$g.png
cp ./plots/$g/.rets.png ./plots/gif/rets$g.png
cp ./plots/$g/.EU.png ./plots/gif/EU$g.png
cp ./plots/$g/.c.png ./plots/gif/c$g.png
done