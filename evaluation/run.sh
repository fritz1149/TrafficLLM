time=$(date +"%Y%m%d%H%M%S")
ptuning_dataset=weixin
dataset_name=qq
sample_num=8000
sampling_method="average_sampling"
model=chatglm2-6b
granularity=packet
checkpoint=12000
export CUDA_VISIBLE_DEVICES=1

nohup python evaluation.py --model_name ~/changc/$model \
                     --test_file ../datasets/changc-${dataset_name}-2025/${sampling_method}-${sample_num}/changc-${dataset_name}-2025_detection_packet_test.json \
                     --label_file ../datasets/changc-${dataset_name}-2025/${sampling_method}-${sample_num}/changc-${dataset_name}-2025_label.json \
                     --traffic_task detection \
                     --ptuning_path ../models/$model/changc-${ptuning_dataset}-2025/${sampling_method}-${sample_num}/checkpoint-$checkpoint \
                     > ../logs/evaluation/$time-$$.txt 2>&1 &