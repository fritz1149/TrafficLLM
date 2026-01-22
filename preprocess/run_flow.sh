#!/bin/bash

granularity="flow"
sample_num=500
dataset_names=("qq weixin USTC-TFC ISCX-VPN-app ISCX-VPN-service CSTNET")

for dataset_name in ${dataset_names[@]}; do
    log_file="../logs/preprocess/$(date +"%Y%m%d%H%M%S")-${dataset_name}-${sample_num}.txt"
    python preprocess_dataset.py \
            --split_mode \
            --input ../../traffic_split/bishe-finetuning/debiased/${dataset_name} \
            --dataset_name ${dataset_name} \
            --traffic_task detection \
            --granularity $granularity \
            --output_path ../datasets/${dataset_name}/${sample_num} \
            --output_name ${dataset_name} > ${log_file} 2>&1 &
done
