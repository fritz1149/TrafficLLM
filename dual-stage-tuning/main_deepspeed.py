#!/usr/bin/env python
# coding=utf-8
"""
DeepSpeed version of main.py with model parallelism (tensor parallelism) for Qwen-VL.
Uses generate + td_evaluation logic for eval and predict.
"""

import logging
import os
import sys
import json
import faulthandler
import signal

import numpy as np
from datasets import load_dataset
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from arguments import ModelArguments, DataTrainingArguments

import deepspeed
from deepspeed import comm as dist

logger = logging.getLogger(__name__)

faulthandler.enable(all_threads=True)
faulthandler.register(signal.SIGUSR1, all_threads=True)


def td_evaluation(predict_responses, target_responses, label_file):
    """
    Evaluation logic from evaluation.py - compute macro metrics.
    """
    with open(label_file, "r", encoding="utf-8") as fin:
        label_dict = json.load(fin)

    preds = []
    labels = []
    for predict_response, target_response in zip(predict_responses, target_responses):
        if ' ' in predict_response:
            predict_response = predict_response.split(" ")[-1]
        labels.append(label_dict[target_response])
        if predict_response in label_dict.keys():
            preds.append(label_dict[predict_response])
        else:
            preds.append(len(label_dict.keys()))

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average='macro', zero_division=0),
        "recall_macro": recall_score(labels, preds, average='macro', zero_division=0),
        "f1_macro": f1_score(labels, preds, average='macro', zero_division=0),
    }
    
    print("acc:", metrics["accuracy"])
    print("precision:", metrics["precision_macro"])
    print("recall:", metrics["recall_macro"])
    print("f1:", metrics["f1_macro"])
    print("confusion matrix:\n", confusion_matrix(labels, preds))
    print("classification report:\n", classification_report(labels, preds, zero_division=0))
    
    return metrics


def get_label_file(data_args):
    """Find label file from test file directory."""
    label_dir = os.path.dirname(data_args.test_file) if data_args.test_file else None
    label_file = None
    if label_dir:
        label_files = [f for f in os.listdir(label_dir) if f.endswith('_label.json')]
        if len(label_files) == 1:
            label_file = os.path.join(label_dir, label_files[0])
        elif len(label_files) > 1:
            raise ValueError(f"Found multiple label files in {label_dir}: {label_files}")
    return label_file


def init_deepspeed_inference(model, mp_size):
    """
    Initialize DeepSpeed inference engine with tensor parallelism.
    """
    ds_config = {
        "tensor_parallel": {"tp_size": mp_size},
        "dtype": "fp16",
        "replace_with_kernel_inject": False,
    }
    
    model = deepspeed.init_inference(
        model,
        mp_size=mp_size,
        dtype=torch.float16,
        replace_with_kernel_inject=False,
    )
    return model


def generate_predictions(model, tokenizer, dataset, data_args, max_new_tokens, device):
    """
    Generate predictions using model.generate().
    Returns list of predicted strings and target strings.
    """
    predictions = []
    targets = []
    
    model.eval()
    
    for i in tqdm(range(len(dataset)), desc="Generating"):
        input_ids = torch.tensor([dataset[i]["input_ids"]], device=device)
        attention_mask = torch.tensor([dataset[i]["attention_mask"]], device=device)
        
        # Get target labels
        label_ids = dataset[i]["labels"]
        # Filter out -100 and pad tokens
        valid_labels = [l for l in label_ids if l != -100]
        target_text = tokenizer.decode(valid_labels, skip_special_tokens=True)
        # Remove <|im_end|> if present
        target_text = target_text.replace("<|im_end|>", "").strip()
        targets.append(target_text)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.7,
                temperature=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode only the generated part (after input)
        generated_ids = outputs[0][input_ids.shape[1]:]
        pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        pred_text = pred_text.replace("<|im_end|>", "").strip()
        predictions.append(pred_text)
    
    return predictions, targets


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
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
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Initialize DeepSpeed distributed
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    logger.warning(
        f"Process rank: {local_rank}, world_size: {world_size}, device: cuda:{local_rank}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    # Load dataset
    data_files = {}
    extension = "json"
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
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    
    # Determine dtype
    qwen_vl_dtype = torch.float16
    if training_args.bf16:
        qwen_vl_dtype = torch.bfloat16
    elif not training_args.fp16:
        if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            qwen_vl_dtype = torch.bfloat16
            training_args.bf16 = True
        else:
            training_args.fp16 = True

    # Load Qwen-VL model
    from transformers import Qwen3VLForConditionalGeneration
    
    if model_args.flash_attn:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=qwen_vl_dtype,
            attn_implementation="flash_attention_2"
        )
        print("flash attn enabled")
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=qwen_vl_dtype,
            low_cpu_mem_usage=True
        )
    
    # Replace visual encoder with Identity (as in original)
    model.visual = nn.Identity()
    
    # Apply PEFT if pre_seq_len is set
    if model_args.pre_seq_len is not None:
        from peft import PrefixTuningConfig, get_peft_model, TaskType
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM, 
            num_virtual_tokens=model_args.pre_seq_len, 
            prefix_projection=model_args.prefix_projection
        )
        model = get_peft_model(model, peft_config)
        model.prompt_encoder.default.float()

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")

    # Initialize DeepSpeed with tensor parallelism for inference
    # For model parallelism, we use DeepSpeed's tensor parallel inference
    mp_size = world_size  # Use all GPUs for model parallelism
    
    device = torch.device(f"cuda:{local_rank}")
    
    # For training, we use DeepSpeed ZeRO Stage 3 which provides model parallelism
    # For inference, we use tensor parallelism
    ds_config = {
        "train_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size,
        "train_micro_batch_size_per_gpu": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": training_args.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": training_args.weight_decay,
                "torch_adam": True,  # Use PyTorch's Adam instead of DeepSpeed's FusedAdam
                "adam_w_mode": True,  # Use AdamW weight decay
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": training_args.learning_rate,
                "warmup_num_steps": training_args.warmup_steps,
            }
        },
        "fp16": {
            "enabled": training_args.fp16,
        },
        "bf16": {
            "enabled": training_args.bf16,
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "none",
            },
            "offload_param": {
                "device": "none",
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e7,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "gradient_clipping": training_args.max_grad_norm,
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }

    # Get pad token id
    QWENVL_PAD_TOKEN = tokenizer.special_tokens_map.get("pad_token", None)
    assert QWENVL_PAD_TOKEN is not None
    QWENVL_PAD_ID = tokenizer.convert_tokens_to_ids([QWENVL_PAD_TOKEN])[0]

    # Preprocessing functions
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    max_target_length = data_args.max_target_length

    def preprocess_function_eval(examples):
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

        return model_inputs

    def preprocess_function_train(examples):
        max_seq_length = data_args.max_source_length + data_args.max_target_length + 1
        model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]

                query = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
                query_ids = tokenizer.encode(query)
                answer = f"{answer}<|im_end|>"
                answer_ids = tokenizer.encode(answer)

                assert len(query_ids) <= data_args.max_source_length, f"query_ids len: {len(query_ids)}"
                assert len(answer_ids) <= data_args.max_target_length, f"answer_ids len: {len(answer_ids)}"
                
                input_ids = query_ids + answer_ids
                pad_len = max_seq_length - len(input_ids)
                origin_len = len(input_ids)
                input_ids = input_ids + [QWENVL_PAD_ID] * pad_len

                attention_mask = [1] * origin_len + [0] * pad_len
                labels = [-100] * len(query_ids) + answer_ids + [-100] * pad_len

                assert len(input_ids) == max_seq_length
                assert len(labels) == max_seq_length
                assert len(attention_mask) == max_seq_length

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)
                model_inputs["attention_mask"].append(attention_mask)

        return model_inputs

    # Prepare datasets
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["eval"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("Nothing to do. Pass `do_train`, `do_eval` or `do_predict`.")
        return

    datasets = {}
    for split_name, preprocess_fn, max_samples in [
        ("train", preprocess_function_train, data_args.max_train_samples),
        ("eval", preprocess_function_eval, data_args.max_eval_samples),
        ("test", preprocess_function_eval, data_args.max_predict_samples),
    ]:
        if split_name in raw_datasets:
            dataset = raw_datasets[split_name]
            if max_samples is not None:
                max_samples = min(len(dataset), max_samples)
                dataset = dataset.select(range(max_samples))
            dataset = dataset.map(
                preprocess_fn,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Processing {split_name} dataset",
            )
            datasets[split_name] = dataset
            if split_name == "train" and local_rank == 0:
                print("=" * 50)
                print("Train example:")
                print("input_ids:", dataset[0]["input_ids"][:50], "...")
                print("inputs:", tokenizer.decode(dataset[0]["input_ids"]))
                print("=" * 50)

    # Initialize DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )

    # Training
    if training_args.do_train:
        logger.info("*** Training ***")
        train_dataset = datasets["train"]
        
        from torch.utils.data import DataLoader, DistributedSampler
        
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
        )
        
        def collate_fn(batch):
            input_ids = torch.tensor([item["input_ids"] for item in batch])
            attention_mask = torch.tensor([item["attention_mask"] for item in batch])
            labels = torch.tensor([item["labels"] for item in batch])
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=0,
        )
        
        num_epochs = int(training_args.num_train_epochs)
        global_step = 0
        
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            model_engine.train()
            epoch_loss = 0.0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=local_rank != 0)
            
            for step, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model_engine(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                
                model_engine.backward(loss)
                model_engine.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                if local_rank == 0:
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_loss = epoch_loss / len(train_dataloader)
            if local_rank == 0:
                logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    # Evaluation and Prediction using generate + td_evaluation
    label_file = get_label_file(data_args)
    max_new_tokens = data_args.max_target_length

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if "eval" not in datasets:
            raise ValueError("Evaluation requires a validation dataset")
        
        eval_dataset = datasets["eval"]
        
        # Only run on rank 0 for generation
        if local_rank == 0:
            # Switch to inference mode
            model_engine.eval()
            
            predictions, targets = generate_predictions(
                model_engine.module if hasattr(model_engine, 'module') else model_engine,
                tokenizer,
                eval_dataset,
                data_args,
                max_new_tokens,
                device,
            )
            
            print("\n*** Evaluation Results ***")
            for pred, target in list(zip(predictions, targets))[:10]:
                print(f"Target: {target}, Predict: {pred}")
            
            if label_file and os.path.exists(label_file):
                metrics = td_evaluation(predictions, targets, label_file)
                
                # Save metrics
                output_dir = training_args.output_dir
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
                    json.dump(metrics, f, indent=2)
            else:
                logger.warning("Label file not found, skipping td_evaluation")
        
        dist.barrier()

    if training_args.do_predict:
        logger.info("*** Predict ***")
        if "test" not in datasets:
            raise ValueError("Prediction requires a test dataset")
        
        test_dataset = datasets["test"]
        
        if local_rank == 0:
            model_engine.eval()
            
            predictions, targets = generate_predictions(
                model_engine.module if hasattr(model_engine, 'module') else model_engine,
                tokenizer,
                test_dataset,
                data_args,
                max_new_tokens,
                device,
            )
            
            print("\n*** Prediction Results ***")
            for pred, target in list(zip(predictions, targets))[:10]:
                print(f"Target: {target}, Predict: {pred}")
            
            # Save predictions
            output_dir = training_args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            output_prediction_file = os.path.join(output_dir, "generated_predictions.txt")
            with open(output_prediction_file, "w", encoding="utf-8") as writer:
                for p, l in zip(predictions, targets):
                    res = json.dumps({"labels": l, "predict": p}, ensure_ascii=False)
                    writer.write(f"{res}\n")
            
            if label_file and os.path.exists(label_file):
                metrics = td_evaluation(predictions, targets, label_file)
                
                with open(os.path.join(output_dir, "predict_results.json"), "w") as f:
                    json.dump(metrics, f, indent=2)
            else:
                logger.warning("Label file not found, skipping td_evaluation")
        
        dist.barrier()

    logger.info("Done!")


if __name__ == "__main__":
    main()
