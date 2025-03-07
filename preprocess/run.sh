granularity=$1

python preprocess_dataset.py --input ~/changc/qq \
                             --dataset_name changc-qq-2025 \
                             --traffic_task detection \
                             --granularity $granularity \
                             --output_path ~/changc/dataset/changc-qq-2025 \
                             --output_name changc-qq-2025