PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1
# export CUDA_VISIBLE_DEVICES=0,1 不默认使用所有显卡，就取消注释
time=$(date +"%Y%m%d%H%M%S")
dataset_name=mixed-etbert-raw
sample_num=8000
model=chatglm2-6b
granularity=packet
export MODEL='chatglm'
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file ../datasets/${dataset_name}/${dataset_name}_detection_${granularity}_train.json \
    --validation_file ../datasets/${dataset_name}/${dataset_name}_detection_${granularity}_test.json \
    --preprocessing_num_workers 10 \
    --prompt_column instruction \
    --response_column output \
    --overwrite_cache \
    --cache_dir ../cache \
    --model_name_or_path ../../$model \
    --output_dir ../models/$model/${dataset_name} \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 32 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 12000 \
    --logging_steps 10 \
    --save_steps 4000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --model_base chatglm \
    > ../logs/train/$time.txt 2>&1