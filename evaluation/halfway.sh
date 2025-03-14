time=$(date +"%Y%m%d%H%M%S")
dataset_name=weixin
sample_num=8000
sampling_method="average_sampling"
model=chatglm2-6b
granularity=packet
transformer_index=10
export CPU=0
export CUDA_VISIBLE_DEVICES=0


echo "************about to preprocess ${dataset_name}, ${sampling_method}, ${sample_num}************"
nohup python halfway.py --model_name ~/changc/$model \
                --test_file ../datasets/changc-${dataset_name}-2025/${sampling_method}-${sample_num}/changc-${dataset_name}-2025_detection_packet_test.json \
                --train_file ../datasets/changc-${dataset_name}-2025/${sampling_method}-${sample_num}/changc-${dataset_name}-2025_detection_packet_train.json \
                --ptuning_path ../models/$model/changc-${dataset_name}-2025/${sampling_method}-${sample_num}/checkpoint-12000 \
                --transformer_index $transformer_index \
                --output_path ../datasets/changc-${dataset_name}-2025/halway-${sampling_method}-${sample_num}-$transformer_index \
                > ../logs/evaluation/$time-$$-${dataset_name}-${sampling_method}-${sample_num}.txt 2>&1 &