#!/bin/bash

RST=rst.txt
declare -a TTI=(100 200 400 600 800 1000)

echo -n > $RST

for f in ${TTI[@]}; do
    echo -e "TTI $f" >> $RST
    cat "$f.csv" >> $RST
    echo "" >> $RST
done