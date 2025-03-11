time=$(date +"%Y%m%d%H%M%S")
nohup python evaluation.py --model_name ~/changc/chatglm2-6b \
                     --test_file ../datasets/changc-weixin-2025/changc-weixin-2025_detection_packet_test.json \
                     --label_file ../datasets/changc-weixin-2025/changc-weixin-2025_label.json \
                     --traffic_task detection \
                     --ptuning_path ../models/chatglm2/changc-qq-weixin-2025/checkpoint-20000 \
                     > ../logs/$time-$$.txt 2>&1 &