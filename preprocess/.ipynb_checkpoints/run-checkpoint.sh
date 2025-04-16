# granularity=$1
# dataset_name=$2
# sameple_num=$3
time=$(date +"%Y%m%d%H%M%S")

granularity="packet"
sameple_nums=(8000)
dataset_names=("mixed-fixed")
sampling_methods=("average_sampling")

for dataset_name in ${dataset_names[@]}; do
    for sampling_method in ${sampling_methods[@]}; do
        for sameple_num in ${sameple_nums[@]}; do
            echo "************about to preprocess ${dataset_name}, ${sampling_method}, ${sameple_num}************"
            python preprocess_dataset.py --input ../../traffic/${dataset_name} \
                                        --dataset_name changc-${dataset_name}-2025 \
                                        --traffic_task detection \
                                        --granularity $granularity \
                                        --output_path ../datasets/changc-${dataset_name}-2025/${sampling_method}-${sameple_num} \
                                        --output_name changc-${dataset_name}-2025 \
                                        --sampling_method $sampling_method \
                                        --max_sampling_number $sameple_num \
                                        >> ../logs/preprocess/$time-$$-${dataset_name}-${sampling_method}-${sameple_num}.txt 2>&1
        done
    done
done