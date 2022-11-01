#!/bin/bash
num1=2000
num2=408

ratio=1.0
echo "4 view " $ratio

bash gen_EP.sh -n 101 -v 4 -o 0 -i 1 -s $(expr $num1*$ratio | bc) -d $(expr $num2*$ratio | bc)
