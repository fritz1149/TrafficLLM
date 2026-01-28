#!/bin/bash
cd dual-stage-tuning

LOG_FILE="../logs/train/$(date +%s)_$$.txt"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'rc=$?; echo "[EXIT] $(date -Is) program finished, exit_code=$rc, log=$LOG_FILE"' EXIT

source /work/miniconda3/etc/profile.d/conda.sh
conda activate trafficllm-qwenvl

export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1

dataset_name=qq
PRE_SEQ_LEN=128
LR=3e-4
export CUDA_VISIBLE_DEVICES=0,1
sample_num=500
max_source_length=4096
model=Qwen3-VL-8B-Instruct
granularity=flow

# 单进程运行，模型按层切分到两张 GPU
python main_model_parallel.py \
    --do_predict \
    --do_train \
    --bf16 \
    --prefix_projection \
    --train_file ../datasets/$dataset_name/$sample_num/$max_source_length/${dataset_name}_detection_${granularity}_train.json \
    --test_file ../datasets/$dataset_name/$sample_num/$max_source_length/${dataset_name}_detection_${granularity}_test.json \
    --preprocessing_num_workers 10 \
    --prompt_column instruction \
    --response_column output \
    --overwrite_cache \
    --cache_dir ../cache \
    --model_name_or_path ../../Bishe_2/$model \
    --output_dir ../models/$model/$dataset_name/$sample_num/$max_source_length \
    --overwrite_output_dir \
    --max_source_length $((max_source_length + 100)) \
    --max_target_length 32 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 10 \
    --logging_steps 1 \
    --save_steps 100 \
    --warmup_ratio 0.1 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --model_base qwen-vl \
    --model_parallel True \
    --model_parallel_split_layer 20 2>&1
