#!/bin/bash
cd ../src

for wal_num in {102..106}
    do
#     wal_num=103

        /opt/MATLAB/R2022a/bin//matlab -nodisplay -nodesktop -nosplash -r "prompt('run');wal_num=['$wal_num']; lvl_num=1;viz_DC_walnut;exit"
#     break
    
    done
    