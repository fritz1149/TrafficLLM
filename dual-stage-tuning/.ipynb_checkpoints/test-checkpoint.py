import torch
import torch.nn as nn
import logging
import warnings
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM, 
    DataCollatorForSeq2Seq,
    Trainer, 
    TrainingArguments,
    HfArgumentParser
)
from peft import (
    LoraConfig, 
    PromptEncoderConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType, 
    PromptEncoderReparameterizationType,
    get_peft_model
)
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers.modeling_utils import PreTrainedModel
import sys
from trainer import PrefixTrainer
from unsloth import FastLanguageModel
# time = datetime.now().strftime("%Y%m%d%H%M%S")

format_style = """### Instruction:
You are an expert with advanced knowledge in understanding network traffic and distinguishing different kinds of traffic.
Please answer the following network traffic identification  question.

### Question:
{}

### Response:
{}
"""

# 设置模型微调的参数类
@dataclass
class FinetuneArguments:
    llm_model_name: str = field(default="deepseek-r1-distill-qwen-7b")
    llm_model_path: str = field(default="../../deepseek-r1-distill-qwen-7b")
    dataset_path: str = field(default="../datasets/changc-mixed-2025/average_sampling-8000/changc-mixed-2025_detection_packet_train.json")
    max_length: int = field(default=1024),
    preprocessing_num_workers: int = field(default=10)
    peft_type: str = field(default="lora")
    lora_rank: int = field(default=8)

def get_alpaca_dataset(json_path: str, test_size: float=0.1):
    dataset = load_dataset(
        'json', 
        data_files=json_path,
        split="train"
    )
    return dataset

def get_tokenizer_dataset(
        dataset, 
        tokenizer,
        num_proc,
    ):

    def process_sample(sample):
        texts = []
        EOS_TOKEN = tokenizer.eos_token
        for instruction, output in zip(sample["instruction"], sample["output"]):
            text = format_style.format(input, output) + EOS_TOKEN
            texts.append(text)
        return {
            "text": texts
        }

    return dataset.map(process_sample, num_proc=num_proc, batched = True)

# 加载LLMs model/tokenizer
def get_base_llm_model_tokenizer(finetune_args):
    # 读取模型类型
    llm_model_name = finetune_args.llm_model_name
    llm_model_path = finetune_args.llm_model_path
    MAX_SEQ_LENGTH = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = llm_model_path,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True
    )
    return model, tokenizer

# 根据peft类型返回相应的config
def get_peft_config(finetune_args, tokenizer):
    # 读取peft类型
    peft_type = finetune_args.peft_type
    print(peft_type)
    if peft_type == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],
            inference_mode=False,
            r=finetune_args.lora_rank,
            lora_alpha=16,
            lora_dropout=0.01,
            bias="none"
        )
    elif peft_type == "p-tuning":
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM, 
            num_virtual_tokens=10,
            encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
            encoder_hidden_size=1024
        )
    elif peft_type == "prefix-tuning":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM, 
            num_virtual_tokens=10,
            prefix_projection=True
        )
    else:
        logger.error("错误参数：peft类型必须为lora/p-tuning/prefix-tuning")
        raise ValueError("错误参数：peft类型必须为lora/p-tuning/prefix-tuning")

    return peft_config

# 微调函数
def finetune_train(model, peft_config, tokenizer, dataset, train_args, finetune_args):
    model = get_peft_model(model=model, peft_config=peft_config)
    model = model.half()
    for k, v in model.named_parameters():
        if v.requires_grad:
            v.float()
    # model.prompt_encoder.default.float()
    # model.gradient_checkpointing_enable()
    print(model)
    model.enable_input_require_grads()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    # trainer = PrefixTrainer(
    #     model=model,
    #     args=train_args,
    #     train_dataset=dataset["train"],
    #     data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    #     save_changed = False
    # )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=False)
    )
    trainer.train()

def main():
    # 忽略警告
    warnings.filterwarnings("ignore")

    # 加载命令行参数
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    # 设置logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # 将logger声明为全局变量
    global logger
    logger = logging.getLogger(__name__)
    logger.debug("命令行参数")
    logger.debug("finetune_args:")
    logger.debug(finetune_args.__repr__())
    logger.debug("training_args:")
    logger.debug(training_args.__repr__())

    # 加载模型
    llm_model, llm_tokenizer = get_base_llm_model_tokenizer(finetune_args)
    logger.info('Base LLMs {} load successfully! LLM path::: {}'.format(finetune_args.llm_model_name, finetune_args.llm_model_path))

    # 获取peft_config参数
    peft_config = get_peft_config(finetune_args, llm_tokenizer)
    logger.info('Peft {} config load successfully!'.format(finetune_args.peft_type))

    # 加载数据
    dataset = get_alpaca_dataset(finetune_args.dataset_path, test_size=0.1)
    logger.info('dataset build successfully!')
    tokenizer_dataset = get_tokenizer_dataset(dataset, llm_tokenizer, max_length=finetune_args.max_length, num_proc=finetune_args.preprocessing_num_workers)
    # tokenizer_dataset = tokenizer_dataset["train"].remove_columns(["text"])
    logger.info('tokenizer dataset build successfully!')

    # 开始训练
    logger.info('Train start!')
    finetune_train(model=llm_model, peft_config=peft_config, tokenizer=llm_tokenizer, dataset=tokenizer_dataset, train_args=training_args, finetune_args=finetune_args)
    logger.info('Train end! Model saves in the path:::{}'.format(training_args.output_dir))


if __name__ == "__main__":
    main()
