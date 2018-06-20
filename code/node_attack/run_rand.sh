#!/bin/bash

dropbox=../../dropbox
dataset=pubmed
gm=gcn
del_rate=0.50
data_folder=$dropbox/data
saved_model=$dropbox/scratch/results/node_classification/$dataset/model-${gm}-epoch-best-${del_rate}

meta_test=0
num_mod=1

python node_rand_attack.py \
    -meta_test $meta_test \
    -num_mod $num_mod \
    -data_folder $data_folder \
    -dataset $dataset \
    -saved_model $saved_model \
    $@
