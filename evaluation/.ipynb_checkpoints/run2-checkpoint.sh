time=$(date +"%Y%m%d%H%M%S")
ptuning_name=$2
dataset_name=$1
num_limit=$3
sampling_method="average_sampling"
model=deepseek-r1-distill-qwen-7b
granularity=packet
checkpoint=12000
# export CUDA_VISIBLE_DEVICES=1
python evaluation.py --model_name ../../$model \
                     --test_file ../datasets/${dataset_name}/${dataset_name}_detection_packet_test.json \
                     --label_file ../datasets/${dataset_name}/${dataset_name}_label.json \
                     --traffic_task detection \
                     --ptuning_path ../models/$model/${dataset_name}/${ptuning_name}/checkpoint-$checkpoint \
                     --num_limit $num_limit \
                     > ../logs/evaluation/$time-$$.txt 2>&1

# run2.sh changc-mixed-2025 