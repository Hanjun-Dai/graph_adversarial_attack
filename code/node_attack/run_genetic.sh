#!/bin/bash

dropbox=../../dropbox
dataset=pubmed

data_folder=$dropbox/data
base_gm=gcn
del_rate=0.01
saved_model=$dropbox/scratch/results/node_classification/$dataset/model-${base_gm}-epoch-best-${del_rate}

lr=0.01
max_lv=1
num_epochs=200
hidden=0
n_hops=0
idx_start=0
num=1000000
pop=50
cross=0.1
mutate=0.2
rounds=5

output_base=$HOME/scratch/results/del_edge_attack/${dataset}-${base_gm}-${del_rate}
save_fold=ga-p-${pop}-c-${cross}-m-${mutate}-r-${rounds}
output_root=$output_base/$save_fold

if [ ! -e $output_root ];
then
    mkdir -p $output_root
fi

python node_genetic.py \
    -data_folder $data_folder \
    -dataset $dataset \
    -idx_start $idx_start \
    -population_size $pop \
    -cross_rate $cross \
    -mutate_rate $mutate \
    -rounds $rounds \
    -num_instances $num \
    -n_hops $n_hops \
    -hidden $hidden \
    -save_dir $output_root \
    -num_epochs $num_epochs \
    -saved_model $saved_model \
    -max_lv $max_lv \
    -learning_rate $lr \
    -instance_id 1968 \
    $@
