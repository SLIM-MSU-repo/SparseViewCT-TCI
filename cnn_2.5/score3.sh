#!/bin/bash

cd /home/gabriel/cnn25/cnn25/cnn25/cnn25_codes/

# export PATH=$PATH:

for fol in {102.npy.mat,103.npy.mat,104.npy.mat,105.npy.mat,106.npy.mat}
do
arr=(${fol//./ })
wal_num=${arr[0]}
matlab -nodisplay -nodesktop -nosplash -r "prompt('run');dir=['cnn25_v4_aug_mask2/matlab_results/'];wal_num=num2str($wal_num);model_name=['$wal_num'];score_walnut;exit"
# break
done


for fol in {102.npy.mat,103.npy.mat,104.npy.mat,105.npy.mat,106.npy.mat}
do
arr=(${fol//./ })
wal_num=${arr[0]}
matlab -nodisplay -nodesktop -nosplash -r "prompt('run');dir=['cnn25_v8_aug_mask2/matlab_results/'];wal_num=num2str($wal_num);model_name=['$wal_num'];score_walnut;exit"
# break
done
