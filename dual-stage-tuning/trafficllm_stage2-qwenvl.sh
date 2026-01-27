PRE_SEQ_LEN=128
LR=2e-2 #TODO: 有待优化
NUM_GPUS=1
# export CUDA_VISIBLE_DEVICES=0,1 不默认使用所有显卡，就取消注释
time=$(date +"%Y%m%d%H%M%S")
dataset_name=$1
sample_num=500
model=Qwen3-VL-8B-Instruct
granularity=flow
export MODEL=qwen

# 创建日志目录
mkdir -p ../logs/train

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ../datasets/$dataset_name/$sample_num/${dataset_name}_detection_${granularity}_train.json \
    --validation_file ../datasets/$dataset_name/$sample_num/${dataset_name}_detection_${granularity}_val.json \
    --test_file ../datasets/$dataset_name/$sample_num/${dataset_name}_detection_${granularity}_test.json \
    --preprocessing_num_workers 10 \
    --prompt_column instruction \
    --response_column output \
    --overwrite_cache \
    --cache_dir ../cache \
    --model_name_or_path ../../Bishe/$model \
    --output_dir ../models/$model/$dataset_name/$sample_num \
    --overwrite_output_dir \
    --max_source_length 5220 \
    --max_target_length 32 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 10 \
    --logging_steps 20 \
    --save_steps 100 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --model_base qwen-vl \
    --remove_unused_columns=False 2>&1 | tee ../logs/train/$time.txt 