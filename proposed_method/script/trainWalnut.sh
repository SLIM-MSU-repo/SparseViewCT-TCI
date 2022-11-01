#!/bin/bash

# example
# bash trainWalnut.sh -d "Walnut101" -v 8 -o 0

while getopts ":d:v:" flag
do
    case "${flag}" in
        d) walnut=${OPTARG};;
        v) nviews=${OPTARG};;
    esac
done

echo "walnut: $walnut";

cd ../src


## Train on Training Walnut
for i in {1..4}
do
   python3 cgan_vol2slice.py --wal_num $walnut --viewtyp $nviews --lvl_num $i --n_epochs 40 # train one stage
   
   python test_CTcgan_v2s_new.py --wal_num $walnut --viewtyp $nviews --lvl_num $i # run one stage trained model on training volume
   
   /opt/MATLAB/R2022a/bin//matlab -nodisplay -nodesktop -nosplash -r "dcCaller($walnut,$i,$nviews);exit" # make output data consistent
done


