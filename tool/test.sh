#!/bin/sh
PARTITION=Segmentation

dataset=$1
model_name=$2
exp_name=$3
exp_dir=exp/${model_name}/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=data/config/${dataset}/${dataset}_${model_name}_${exp_name}.yaml

mkdir -p ${model_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp tool/test.sh tool/test.py ${config} ${exp_dir}

python -u -m tool.test --config=${config} 2>&1 | tee ${result_dir}/test-$now.log
