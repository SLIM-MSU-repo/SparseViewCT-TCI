#!/bin/bash
while getopts ":n:v:o:i:s:d:" flag
do
    case "${flag}" in
        n) wal_num=${OPTARG};;
        v) nviews=${OPTARG};;
	o) obt_strt=${OPTARG};;
	i) in=${OPTARG};;
    s) dist_sd=${OPTARG};;
    d) dist_od=${OPTARG};;
    esac
done
echo "Walnut Num: $wal_num";
echo "Views: $nviews";
echo "Rotation: $obt_strt";
echo "Initialization: $in";
echo "dsd: $dist_sd";
echo "dod: $dist_od";

cd ../src/
# for sig in {6..9..3}

sig=10; # sigma used for data augmentation

dir='../../data_dir/Walnut'$wal_num'/'
echo $dir


# generate 8 view EP reconstructions
for num in {101,102,103,104,105,106}
    do
    matlab -nodisplay -nodesktop -nosplash -r "prompt('run');init_EP_2($wal_num,8,$obt_strt,$dist_sd,$dist_od,$sig,$num,['$dir']);exit;"
    done

# generate 4 view EP reconstructions
for num in {101,102,103,104,105,106}
    do
    matlab -nodisplay -nodesktop -nosplash -r "prompt('run');init_EP_2($wal_num,4,$obt_strt,$dist_sd,$dist_od,$sig,$num,['$dir']);exit;"
    done