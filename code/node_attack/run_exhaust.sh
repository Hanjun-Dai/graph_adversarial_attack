#!/bin/bash

dropbox=../../dropbox
dataset=pubmed
gm=gcn
n_hops=0
del_rate=0.01
data_folder=$dropbox/data
saved_model=$dropbox/scratch/results/node_classification/$dataset/model-${gm}-epoch-best-${del_rate}
meta_test=0

output_base=$HOME/scratch/results/del_edge_attack/
save_fold=${dataset}-${gm}-${del_rate}

output_root=$output_base/$save_fold/exhaust-meta-${meta_test}

if [ ! -e $output_root ];
then
    mkdir -p $output_root
fi

python exhaust_attack.py \
    -meta_test $meta_test \
    -data_folder $data_folder \
    -n_hops $n_hops \
    -dataset $dataset \
    -save_dir $output_root \
    -saved_model $saved_model \
    $@
