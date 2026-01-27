<<<<<<< HEAD
cd dual-stage-tuning

LOG_FILE="../logs/train/$(date +%s)_$$.txt"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'rc=$?; echo "[EXIT] $(date -Is) rag-corpus-index finished, exit_code=$rc, log=$LOG_FILE"' EXIT
=======
dataset_name=qq;
PRE_SEQ_LEN=128;
LR=2e-2 #TODO: 有待优化;
NUM_GPUS=1;
# export CUDA_VISIBLE_DEVICES=0,1 不默认使用所有显卡，就取消注释;
time=$(date +"%Y%m%d%H%M%S");
sample_num=500;
model=Qwen3-VL-8B-Instruct;
granularity=flow;
export MODEL=qwen;

# 创建日志目录;
mkdir -p ../logs/train;
>>>>>>> a8be7ed (1)

source /work/miniconda3/etc/profile.d/conda.sh
conda activate trafficllm-qwenvl

dataset_name=qq;
PRE_SEQ_LEN=128;
LR=2e-2 #TODO: 有待优化;
NUM_GPUS=1;
# export CUDA_VISIBLE_DEVICES=0,1 不默认使用所有显卡，就取消注释;
time=$(date +"%Y%m%d%H%M%S");
sample_num=500;
model=Qwen3-VL-8B-Instruct;
granularity=flow;
export MODEL=qwen;

python main.py \
    --do_train \
    --do_predict \
    --bf16 \
    --flash_attn \
<<<<<<< HEAD
    --model_parallel \
    --model_parallel_split_layer 18 \
=======
>>>>>>> a8be7ed (1)
    --predict_with_generate \
    --train_file ../datasets/$dataset_name/$sample_num/${dataset_name}_detection_${granularity}_train.json \
    --test_file ../datasets/$dataset_name/$sample_num/${dataset_name}_detection_${granularity}_test.json \
    --preprocessing_num_workers 10 \
    --prompt_column instruction \
    --response_column output \
    --overwrite_cache \
    --cache_dir ../cache \
    --model_name_or_path ../../Bishe_2/$model \
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