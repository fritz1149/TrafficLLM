models=(chatglm2-6b DeepSeek-R1-Distill-Qwen-7B)
model=${models[1]}
dataset_name=qq
sampling_method=average_sampling
sample_num=8000
python test.py --model_name ~/changc/$model \
               --ptuning_path ../models/$model/changc-${dataset_name}-2025/${sampling_method}-${sample_num}/checkpoint-12000 
# echo ${models[1]}