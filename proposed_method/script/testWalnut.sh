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
for wal_num in {102..106..1}
do
echo $wal_num
    for i in {1..4}
    do
    echo 'stage' $i
        python test_CTcgan_v2s_new.py --wal_num $wal_num --viewtyp $nviews --orbit_start 0.0 --lvl_num $i #--batch_size 8
        /opt/MATLAB/R2022a/bin//matlab -nodisplay -nodesktop -nosplash -r "dcCaller($wal_num,$i,$nviews);exit"
    done
done

