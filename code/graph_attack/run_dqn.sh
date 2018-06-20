#!/bin/bash

dropbox=../../dropbox

min_n=90
max_n=100
p=0.02
min_c=1
max_c=3
base_lv=2
data_folder=$dropbox/data/components
save_fold=nodes-${min_n}-${max_n}-p-${p}-c-${min_c}-${max_c}-lv-${base_lv}
base_model_dump=$dropbox/scratch/results/graph_classification/components/$save_fold/epoch-best

lr=0.001
max_lv=5
frac_meta=0.1

output_base=$HOME/scratch/results/graph_classification/components/$save_fold

output_root=$output_base/lv-${max_lv}-frac-${frac_meta}

if [ ! -e $output_root ];
then
    mkdir -p $output_root
fi

python dqn.py \
    -data_folder $data_folder \
    -save_dir $output_root \
    -max_n $max_n \
    -min_n $min_n \
    -max_lv $max_lv \
    -frac_meta $frac_meta \
    -min_c $min_c \
    -max_c $max_c \
    -n_graphs 5000 \
    -er_p $p \
    -learning_rate $lr \
    -base_model_dump $base_model_dump \
    -logfile $output_root/log.txt \
    $@
