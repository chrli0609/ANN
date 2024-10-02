cd png/$1
convert -delay 50 -loop 1 $(ls *.png | sort -n) ../../gif/$1.gif
