time=$(date +"%Y%m%d%H%M%S")
export MODEL='qwen'
python test.py \
    --peft_type lora \
    --lora_rank 8 \
    --llm_model_name deepseek-r1-distill-qwen-7b \
    --llm_model_path ../../deepseek-r1-distill-qwen-7b \
    --dataset_path ../datasets/changc-mixed-2025/average_sampling-8000/changc-mixed-2025_detection_packet_train.json \
    --max_length 1024 \
    --output_dir ../models/deepseek-r1-distill-qwen-7b/test-lora \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 1 \
    --learning_rate 2e-2 \
    --max_steps 20 \
    --save_steps 20 \
    --logging_steps 10 \
    --overwrite_output_dir \
    --gradient_accumulation_steps 16 \
    --remove_unused_columns=False \
    > ../logs/train/$time.txt 2>&1

