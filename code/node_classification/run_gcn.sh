#!/bin/bash

dropbox=../../dropbox
dataset=pubmed

data_folder=$dropbox/data
gm=gcn
del_rate=0.00
output_root=$dropbox/scratch/results/node_classification/$dataset

lr=0.01
max_lv=2
num_epochs=200
hidden=0

python gcn.py \
    -data_folder $data_folder \
    -del_rate $del_rate \
    -dataset $dataset \
    -gm $gm \
    -hidden $hidden \
    -save_dir $output_root \
    -num_epochs $num_epochs \
    -max_lv $max_lv \
    -learning_rate $lr \
    $@
