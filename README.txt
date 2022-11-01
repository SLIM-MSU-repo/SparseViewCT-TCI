To run 2.5D model or the proposed approach you need to first generate EP reconstructions of your dataset. In order to do so run 

proposed_method/script/runfdk_ep.sh

This script allows you to generate EP reconstructions from sparse views with input of source-detector and object-detector distances along with number of views. You may have to modify the above script to point to the right path of your dataset.


After running and saving the EP reconstructions for four and eight views, you can run the below - 

2.5D CNN 
go to cnn_2.5/

1. bash run_all.sh # this will run the 2.5D training on 4 view and 8 view cases.
2. bash score3.sh # this will obtain the NMAE and NHFEN scores for the 2.5 D CNN outputs.

Proposed approach 

go to proposed_method/script/ and run 

1. bash trainWalnut.sh # this trains the proposed approach.
2. bash testWalnut.sh # this tests the proposed approach on unseen cases.
3. bash get_metrics.sh # this gets the NMAE and HFEN scores
