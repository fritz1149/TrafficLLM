# granularity=$1
# dataset_name=$2
# sameple_num=$3
time=$(date +"%Y%m%d%H%M%S")

granularity="packet"
sameple_num=1000
dataset_names=("qq" "weixin" "qq-weixin")

for dataset_name in ${dataset_names[@]}; do
    echo "************about to preprocess ${dataset_name}************"
    python preprocess_dataset.py --input ~/changc/traffic/${dataset_name} \
                                --dataset_name changc-${dataset_name}-2025 \
                                --traffic_task detection \
                                --granularity $granularity \
                                --output_path ../datasets/changc-${dataset_name}-2025/$sameple_num \
                                --output_name changc-${dataset_name}-2025 \
                                >> ../logs/preprocess/$time-$$txt 2>&1
done