#!/bin/bash

for g in 0 1 2 3 4 5 6 7 8
do
#echo "$g"
cp ./plots/$g/.e.png ./plots/gif/eq$g.png
cp ./plots/$g/.b.png ./plots/gif/b$g.png
cp ./plots/$g/.exret.png ./plots/gif/exret$g.png
cp ./plots/$g/.rets.png ./plots/gif/rets$g.png
cp ./plots/$g/.EU.png ./plots/gif/EU$g.png
cp ./plots/$g/.c.png ./plots/gif/c$g.png
cp ./plots/$g/.p.png ./plots/gif/p$g.png
cp ./plots/$g/.q.png ./plots/gif/q$g.png
cp ./plots/$g/.plot_losses.png ./plots/gif/losses$g.png
done

for plot in eq b exret rets EU c p q losses
do
convert -delay 75 -loop 0 ./plots/gif/$plot*.png -scale 1920x1080 ./plots/gif/$plot.gif
done

rm ./plots/gif/*.png