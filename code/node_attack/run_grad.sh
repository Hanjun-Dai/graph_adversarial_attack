#!/bin/bash

dropbox=../../dropbox
dataset=cora
gm=gcn
n_hops=2
del_rate=0.00
data_folder=$dropbox/data
saved_model=$dropbox/scratch/results/node_classification/$dataset/model-${gm}-epoch-best-${del_rate}

targeted=0
num_mod=1
output_base=$HOME/scratch/results/del_edge_attack/

save_fold=${dataset}-${gm}

output_root=$output_base/$save_fold/grad-t-${targeted}-m-${num_mod}


if [ ! -e $output_root ];
then
    mkdir -p $output_root
fi

python node_grad_attack.py \
    -num_mod $num_mod \
    -targeted $targeted \
    -data_folder $data_folder \
    -n_hops $n_hops \
    -dataset $dataset \
    -save_dir $output_root \
    -saved_model $saved_model \
    $@
