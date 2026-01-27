#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import json
import importlib.util

import numpy as np
from datasets import load_dataset
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    set_seed,
    Trainer
)
from arguments import ModelArguments, DataTrainingArguments

logger = logging.getLogger(__name__)

def main():
    # if os.environ['MODEL'] == 'qwen':
    #     parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # elif os.environ['MODEL'] == 'chatglm':
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load dataset
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["eval"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        # use_auth_token=True if model_args.use_auth_token else None,
    )
    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    qwen_vl_torch_dtype = None
    if model_args.model_base == 'qwen-vl' and (not training_args.fp16) and (not training_args.bf16):
        if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            training_args.bf16 = True
        else:
            training_args.fp16 = True
    if model_args.model_base == 'qwen-vl':
        if training_args.bf16:
            qwen_vl_torch_dtype = torch.bfloat16
        elif training_args.fp16:
            qwen_vl_torch_dtype = torch.float16
    
    if model_args.model_base == 'qwen':
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
        from peft import PrefixTuningConfig, get_peft_model, TaskType
        peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=model_args.pre_seq_len)
        model = get_peft_model(model, peft_config)
    elif model_args.model_base == 'qwen-vl':
        from transformers import Qwen3VLForConditionalGeneration
        if model_args.flash_attn:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=qwen_vl_torch_dtype,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2"
            )
            print("flash attn")
            print("attn_implementation =", getattr(model.config, "attn_implementation", None))
            import sys
            sys.stdout.flush()
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=qwen_vl_torch_dtype,
                low_cpu_mem_usage=True
            )
        from peft import PrefixTuningConfig, get_peft_model, TaskType
        peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=model_args.pre_seq_len, prefix_projection=model_args.prefix_projection)
        model = get_peft_model(model, peft_config)
        QWENVL_PAD_TOKEN = tokenizer.special_tokens_map.get("pad_token", None)
        assert QWENVL_PAD_TOKEN is not None
        QWENVL_PAD_ID = tokenizer.convert_tokens_to_ids([QWENVL_PAD_TOKEN])[0]
    elif model_args.model_base == 'chatglm':
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        config.pre_seq_len = model_args.pre_seq_len
        config.prefix_projection = model_args.prefix_projection
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    print("trainable parameters: ", end="")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    # model.print_trainable_parameters()
    # print(model.active_peft_config.inference_mode)
    # print(model)
    # return

    # TODO：待修改或添加（若用到）
    if model_args.ptuning_checkpoint is not None:
        # Evaluation
        # Loading extra state dict of prefix encoder
        prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        if not model_args.model_base.startswith('qwen'):
            model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        else:
            model.prompt_encoder.default.load_state_dict(new_prefix_state_dict)

    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)
    # TODO：待修改或添加（若报错）
    if model_args.pre_seq_len is not None:
        # P-tuning v2
        if model_args.model_base == 'qwen-vl':
            model.prompt_encoder.default.float()
        else:
            model = model.half()
            if not model_args.model_base.startswith('qwen'):
                model.transformer.prefix_encoder.float()
            else:
                model.prompt_encoder.default.float()
    else:
        # Finetune
        if model_args.model_base != 'qwen-vl':
            model = model.float()

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["eval"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    history_column = data_args.history_column
    
    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length

    def preprocess_function_eval(examples):
        if model_args.model_base == 'qwen-vl':
            model_inputs = {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }
            for i in range(len(examples[prompt_column])):
                if examples[prompt_column][i] and examples[response_column][i]:
                    query, answer = examples[prompt_column][i], examples[response_column][i]

                    query = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
                    input_ids = tokenizer.encode(query)
                    origin_len = len(input_ids)
                    assert origin_len <= data_args.max_source_length
                    pad_len = data_args.max_source_length - origin_len
                    input_ids = [QWENVL_PAD_ID] * pad_len + input_ids
                    attention_mask = [0] * pad_len + [1] * origin_len

                    answer = f"{answer}<|im_end|>"
                    labels = tokenizer.encode(answer)
                    assert len(labels) <= max_target_length
                    pad_len = max_target_length - len(labels)
                    labels = labels + [QWENVL_PAD_ID] * pad_len

                    model_inputs["input_ids"].append(input_ids)
                    model_inputs["attention_mask"].append(attention_mask)
                    model_inputs["labels"].append(labels)
        elif model_args.model_base == 'chatglm':
            inputs, targets = [], []
            for i in range(len(examples[prompt_column])):
                if examples[prompt_column][i] and examples[response_column][i]:
                    query = examples[prompt_column][i]
                    history = examples[history_column][i] if history_column is not None else None
                    prompt = tokenizer.build_prompt(query, history)
                    inputs.append(prompt)
                    targets.append(examples[response_column][i])

            inputs = [prefix + inp for inp in inputs]
            model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, truncation=True, padding=True)
            labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

            if data_args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]
            model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def preprocess_function_train(examples):
        max_seq_length = data_args.max_source_length + data_args.max_target_length + 1
        
        if model_args.model_base.startswith('qwen'):
            model_inputs = {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }
        else:
            model_inputs = {
                "input_ids": [],
                "labels": [],
            }
        if model_args.model_base == 'qwen':
            for i in range(len(examples[prompt_column])):
                if examples[prompt_column][i] and examples[response_column][i]:
                    query, answer = examples[prompt_column][i], examples[response_column][i]
                    prompt = f"[Round 1]\n\n问：{query}\n\n答："
                    prompt = prefix + prompt
                    instruction = tokenizer(prompt)
                    response = tokenizer(answer + tokenizer.eos_token)
                    input_ids = instruction["input_ids"] + response["input_ids"]
                    attention_mask = instruction["attention_mask"] + response["attention_mask"]
                    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
                    if len(input_ids) > data_args.max_source_length:
                        input_ids = input_ids[:data_args.max_source_length]
                        attention_mask = attention_mask[:data_args.max_source_length]
                        labels = labels[:data_args.max_source_length]
                    model_inputs["input_ids"].append(input_ids)
                    model_inputs["labels"].append(labels)
                    model_inputs["attention_mask"].append(attention_mask)
        if model_args.model_base == 'qwen-vl':
            for i in range(len(examples[prompt_column])):
                if examples[prompt_column][i] and examples[response_column][i]:
                    query, answer = examples[prompt_column][i], examples[response_column][i]

                    query = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
                    query_ids = tokenizer.encode(query)
                    answer = f"{answer}<|im_end|>"
                    answer_ids = tokenizer.encode(answer)

                    assert len(query_ids) <= data_args.max_source_length, f"query_ids len: {len(query_ids)}"
                    assert len(answer_ids) <= data_args.max_target_length, f"answer_ids len: {len(answer_ids)}"
                    
                    input_ids = query_ids+answer_ids
                    pad_len = max_seq_length - len(input_ids)
                    origin_len = len(input_ids)
                    input_ids = input_ids + [QWENVL_PAD_ID] * pad_len

                    attention_mask = [1] * origin_len + [0] * pad_len
                    labels = [-100] * len(query_ids) + answer_ids + [-100] * pad_len

                    assert len(input_ids) == max_seq_length, f"input_ids len: {len(input_ids)}"
                    assert len(labels) == max_seq_length, f"labels len: {len(labels)}"
                    assert len(attention_mask) == max_seq_length, f"attention_mask len: {len(attention_mask)}"

                    model_inputs["input_ids"].append(input_ids)
                    model_inputs["labels"].append(labels)
                    model_inputs["attention_mask"].append(attention_mask)

        elif model_args.model_base == 'chatglm':
            for i in range(len(examples[prompt_column])):
                if examples[prompt_column][i] and examples[response_column][i]:
                    query, answer = examples[prompt_column][i], examples[response_column][i]

                    history = examples[history_column][i] if history_column is not None else None
                    prompt = tokenizer.build_prompt(query, history)

                    prompt = prefix + prompt
                    a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,
                                            max_length=data_args.max_source_length)
                    b_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,
                                            max_length=data_args.max_target_length)

                    context_length = len(a_ids)
                    input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
                    labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]
                    
                    pad_len = max_seq_length - len(input_ids)
                    input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                    labels = labels + [tokenizer.pad_token_id] * pad_len
                    if data_args.ignore_pad_token_for_loss:
                        labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

                    model_inputs["input_ids"].append(input_ids)
                    model_inputs["labels"].append(labels)

        return model_inputs
    
    def print_dataset_example(example):
        print("=" * 50)
        print("input_ids", example["input_ids"])
        print("inputs", tokenizer.decode(example["input_ids"]))
        print("label_ids", example["labels"])
        filtered_labels = [l for l in example["labels"] if l != -100]
        print("labels", tokenizer.decode(filtered_labels))
        print("=" * 50)

    def prepare_dataset(
        split_name,
        preprocess_fn,
        max_samples,
        stage_desc,
        map_desc,
        require_message,
        print_example=False,
    ):
        if split_name not in raw_datasets:
            raise ValueError(require_message)
        dataset = raw_datasets[split_name]
        if max_samples is not None:
            max_samples = min(len(dataset), max_samples)
            dataset = dataset.select(range(max_samples))
        with training_args.main_process_first(desc=stage_desc):
            dataset = dataset.map(
                preprocess_fn,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=map_desc,
            )
        if print_example:
            print_dataset_example(dataset[0])
        return dataset

    preprocess_functions = {
        "train": preprocess_function_train,
        "eval": preprocess_function_eval,
        "test": preprocess_function_eval,
    }
    max_samples = {
        "train": data_args.max_train_samples,
        "eval": data_args.max_eval_samples,
        "test": data_args.max_predict_samples,
    }
    dataset = {}
    for split_name in ["train", "eval", "test"]:
        if split_name in raw_datasets:
            dataset[split_name] = prepare_dataset(
                split_name=split_name,
                preprocess_fn=preprocess_functions[split_name],
                max_samples=max_samples[split_name],
                stage_desc=f"{split_name} dataset map pre-processing",
                map_desc=f"Running tokenizer on {split_name} dataset",
                require_message=f"--do_{split_name} requires a {split_name} dataset",
                print_example=(split_name == "train"),
            )
    if training_args.do_train:
        train_dataset = dataset["train"]
    if training_args.do_eval:
        eval_dataset = dataset["eval"]
    if training_args.do_predict:
        test_dataset = dataset["test"]

    # Metric
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
            result = scores[0]
            
            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    # Initialize our Trainer
    # if model_args.model_base == 'chatglm':
        # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )
    from trainer_seq2seq import Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
            pad_to_multiple_of=None,
            padding=False
        ),
        # compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        save_changed=model_args.pre_seq_len is not None
    )
    # elif model_args.model_base == 'qwen':
    #     from trainer import PrefixTrainer
    #     if torch.cuda.device_count() > 1:
    #         print("Let's use", torch.cuda.device_count(), "GPUs!")
    #         model = nn.DataParallel(model)
    #     trainer = PrefixTrainer(
    #         model=model,
    #         args=training_args,
    #         train_dataset=train_dataset,
    #         tokenizer=tokenizer,
    #         data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    #         # save_changed=model_args.pre_seq_len is not None
    #         save_changed=False
    #     )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        if model_args.model_base == 'qwen-vl':
            pass
        else:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        if model_args.model_base == 'qwen-vl':
            pass
        else:
            model.gradient_checkpointing_disable()
            model.disable_input_require_grads()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    # Evaluation
    results = {}
    max_seq_length = data_args.max_source_length + data_args.max_target_length + 1
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval", do_sample=True, top_p=0.7, max_length=max_seq_length, temperature=0.95)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(test_dataset, metric_key_prefix="predict", max_length=max_seq_length, do_sample=True, top_p=0.7, temperature=0.95)
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(test_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(test_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                label_ids = predict_results.label_ids
                labels = tokenizer.batch_decode(
                    label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                labels = [label.strip() for label in labels]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    for p, l in zip(predictions, labels):
                        res = json.dumps({"labels": l, "predict": p}, ensure_ascii=False)
                        writer.write(f"{res}\n")

                label_dir = os.path.dirname(data_args.test_file) if data_args.test_file else None
                label_file = None
                if label_dir:
                    label_files = [f for f in os.listdir(label_dir) if f.endswith('_label.json')]
                    if len(label_files) == 1:
                        label_file = os.path.join(label_dir, label_files[0])
                    elif len(label_files) > 1:
                        raise ValueError(f"Found multiple label files in {label_dir}: {label_files}")
                if label_file is not None and os.path.exists(label_file):
                    try:
                        eval_py = os.path.abspath(
                            os.path.join(os.path.dirname(__file__), "..", "evaluation", "evaluation.py")
                        )
                        spec = importlib.util.spec_from_file_location("trafficllm_evaluation", eval_py)
                        evaluation_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(evaluation_module)
                        evaluation_module.td_evaluation(predictions, labels, label_file)
                    except Exception as e:
                        logger.warning(f"td_evaluation failed: {e}")
                else:
                    logger.warning("LABEL_FILE env var is not set or path does not exist; skip td_evaluation")

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
