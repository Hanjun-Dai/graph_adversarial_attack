#!/bin/bash

dropbox=../../dropbox
data_folder=$dropbox/data

dataset=pubmed
base_gm=gcn
del_rate=0.01
saved_model=$dropbox/scratch/results/node_classification/$dataset/model-${base_gm}-epoch-best-${del_rate}

lr=0.01
max_lv=1
num_epochs=200
batch_size=10
hidden=0
n_hops=3
bilin_q=1
reward_type=binary
gm=mean_field
adj_norm=1
meta_test=0
num_mod=1

output_base=$HOME/scratch/results/del_edge_attack/${dataset}-${base_gm}-${del_rate}
save_fold=rl-lv-${max_lv}-q-${bilin_q}-meta-${meta_test}
output_root=$output_base/$save_fold

if [ ! -e $output_root ];
then
    mkdir -p $output_root
fi

#export CUDA_VISIBLE_DEVICES=1
python node_dqn.py \
    -meta_test $meta_test \
    -num_mod $num_mod \
    -data_folder $data_folder \
    -dataset $dataset \
    -reward_type $reward_type \
    -bilin_q $bilin_q \
    -n_hops $n_hops \
    -gm $gm \
    -adj_norm $adj_norm \
    -hidden $hidden \
    -batch_size $batch_size \
    -save_dir $output_root \
    -num_epochs $num_epochs \
    -saved_model $saved_model \
    -max_lv $max_lv \
    -learning_rate $lr \
    -num_steps 500000 \
    -phase train \
    $@
