models=("chatglm2-6b" "deepseek-r1-distill-qwen-7b")
model=chatglm2-6b
dataset_name=qq
sampling_method=average_sampling
sample_num=8000
python test.py --model_name ../../$model
# echo $model