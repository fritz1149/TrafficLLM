#!/usr/bin/env python
# coding=utf-8
"""
Model parallel version using accelerate dispatch_model.
Uses layer-wise device placement across 2 GPUs.
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
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    get_cosine_schedule_with_warmup,
)
from accelerate import dispatch_model
from arguments import ModelArguments, DataTrainingArguments

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


def create_device_map(model_args, num_hidden_layers=36):
    """
    Create device map for layer-wise model parallelism across 2 GPUs.
    Similar to main.py logic.
    """
    split_layer = getattr(model_args, "model_parallel_split_layer", None)
    if split_layer is None:
        split_layer = num_hidden_layers // 2
    
    device_map = {
        "base_model.model.language_model.embed_tokens": "cuda:0",
        "base_model.model.language_model.rotary_emb": "cuda:0",
        "base_model.model.visual": "cuda:0",
        "word_embeddings": "cuda:0",
        "prompt_encoder": "cuda:0",
        "base_model.lm_head": "cuda:1",
        "base_model.model.language_model.norm": "cuda:1",
        **{f"base_model.model.language_model.layers.{i}": "cuda:0" for i in range(0, split_layer)},
        **{f"base_model.model.language_model.layers.{i}": "cuda:1" for i in range(split_layer, num_hidden_layers)},
    }
    return device_map, split_layer


def patch_get_prompt_for_mp(model, split_layer):
    """
    Patch model.get_prompt to move past_key_values to correct devices.
    """
    if not hasattr(model, "get_prompt"):
        return
    
    _orig_get_prompt = model.get_prompt

    def _mp_get_prompt(batch_size, max_cache_len):
        past_key_values = _orig_get_prompt(batch_size, max_cache_len)
        if past_key_values is None:
            return past_key_values

        # transformers DynamicCache-style object
        assert hasattr(past_key_values, "layers") and isinstance(getattr(past_key_values, "layers"), list)

        for i, layer_cache in enumerate(past_key_values.layers):
            dev = torch.device("cuda:0" if i < split_layer else "cuda:1")
            if hasattr(layer_cache, "device"):
                layer_cache.device = dev
            if hasattr(layer_cache, "keys") and torch.is_tensor(layer_cache.keys) and layer_cache.keys.device != dev:
                layer_cache.keys = layer_cache.keys.to(dev)
            if hasattr(layer_cache, "values") and torch.is_tensor(layer_cache.values) and layer_cache.values.device != dev:
                layer_cache.values = layer_cache.values.to(dev)
        return past_key_values

    model.get_prompt = _mp_get_prompt


def generate_predictions(model, tokenizer, dataset, data_args, max_new_tokens, batch_size=1):
    """
    Generate predictions using model.generate().
    """
    predictions = []
    targets = []
    
    model.eval()
    
    def collate_fn(batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch])
        attention_mask = torch.tensor([item["attention_mask"] for item in batch])
        labels = [item["labels"] for item in batch]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    for batch in tqdm(dataloader, desc="Generating"):
        input_ids = batch["input_ids"].to("cuda:0")
        attention_mask = batch["attention_mask"].to("cuda:0")

        for label_ids in batch["labels"]:
            valid_labels = [l for l in label_ids if l != -100]
            target_text = tokenizer.decode(valid_labels, skip_special_tokens=True)
            target_text = target_text.replace("<|im_end|>", "").strip()
            targets.append(target_text)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.7,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i in range(outputs.shape[0]):
            generated_ids = outputs[i][input_ids.shape[1]:]
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

    logger.warning(f"device: cuda, n_gpu: {torch.cuda.device_count()}")
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    # Check GPU availability
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        raise ValueError("Model parallel requires at least 2 CUDA devices")
    
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
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )
        print("flash attn enabled")
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=qwen_vl_dtype,
            low_cpu_mem_usage=True,
        )
    
    # Replace visual encoder with Identity
    model.visual = nn.Identity()
    
    # Apply PEFT prefix tuning
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

    # Get number of hidden layers
    num_hidden_layers = 36
    
    # Create device map and dispatch model
    device_map, split_layer = create_device_map(model_args, num_hidden_layers)
    print(f"Using model parallel with split_layer={split_layer}, num_hidden_layers={num_hidden_layers}")
    print(f"Device map: {device_map}")
    
    model = dispatch_model(model, device_map=device_map)
    
    # Patch get_prompt for model parallel
    patch_get_prompt_for_mp(model, split_layer)

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
            if split_name == "train":
                print("=" * 50)
                print("Train example:")
                print("input_ids:", dataset[0]["input_ids"][:50], "...")
                print("inputs:", tokenizer.decode(dataset[0]["input_ids"]))
                print("=" * 50)

    def run_eval_predict(epoch_idx=None):
        label_file = get_label_file(data_args)
        max_new_tokens = data_args.max_target_length
        epoch_suffix = f"_epoch_{epoch_idx}" if epoch_idx is not None else ""

        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            if "eval" not in datasets:
                raise ValueError("Evaluation requires a validation dataset")

            eval_dataset = datasets["eval"]

            predictions, targets = generate_predictions(
                model,
                tokenizer,
                eval_dataset,
                data_args,
                max_new_tokens,
                batch_size=training_args.per_device_eval_batch_size,
            )

            print("\n*** Evaluation Results ***")
            for pred, target in list(zip(predictions, targets))[:10]:
                print(f"Target: {target}, Predict: {pred}")

            if label_file and os.path.exists(label_file):
                metrics = td_evaluation(predictions, targets, label_file)

                output_dir = training_args.output_dir
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, f"eval_results{epoch_suffix}.json"), "w") as f:
                    json.dump(metrics, f, indent=2)
            else:
                logger.warning("Label file not found, skipping td_evaluation")

        if training_args.do_predict:
            logger.info("*** Predict ***")
            if "test" not in datasets:
                raise ValueError("Prediction requires a test dataset")

            test_dataset = datasets["test"]

            predictions, targets = generate_predictions(
                model,
                tokenizer,
                test_dataset,
                data_args,
                max_new_tokens,
                batch_size=training_args.per_device_eval_batch_size,
            )

            print("\n*** Prediction Results ***")
            for pred, target in list(zip(predictions, targets))[:10]:
                print(f"Target: {target}, Predict: {pred}")

            output_dir = training_args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            output_prediction_file = os.path.join(output_dir, f"generated_predictions{epoch_suffix}.txt")
            with open(output_prediction_file, "w", encoding="utf-8") as writer:
                for p, l in zip(predictions, targets):
                    res = json.dumps({"labels": l, "predict": p}, ensure_ascii=False)
                    writer.write(f"{res}\n")

            if label_file and os.path.exists(label_file):
                metrics = td_evaluation(predictions, targets, label_file)

                with open(os.path.join(output_dir, f"predict_results{epoch_suffix}.json"), "w") as f:
                    json.dump(metrics, f, indent=2)
            else:
                logger.warning("Label file not found, skipping td_evaluation")

    # Training
    if training_args.do_train:
        logger.info("*** Training ***")
        train_dataset = datasets["train"]
        
        def collate_fn(batch):
            input_ids = torch.tensor([item["input_ids"] for item in batch])
            attention_mask = torch.tensor([item["attention_mask"] for item in batch])
            labels = torch.tensor([item["labels"] for item in batch])
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )
        
        # Setup optimizer - only optimize trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
        
        # Setup scheduler
        num_training_steps = len(train_dataloader) * int(training_args.num_train_epochs) // training_args.gradient_accumulation_steps
        num_warmup_steps = training_args.warmup_steps if training_args.warmup_steps > 0 else int(num_training_steps * training_args.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        
        num_epochs = int(training_args.num_train_epochs)
        global_step = 0
        
        # Mixed precision scaler
        scaler = torch.amp.GradScaler('cuda') if training_args.fp16 else None
        use_amp = training_args.fp16 or training_args.bf16
        amp_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Move to first GPU (dispatch_model will handle cross-GPU)
                input_ids = batch["input_ids"].to("cuda:0")
                attention_mask = batch["attention_mask"].to("cuda:0")
                labels = batch["labels"].to("cuda:1")  # Labels should be on same device as lm_head output
                
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss = outputs.loss / training_args.gradient_accumulation_steps
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / training_args.gradient_accumulation_steps
                
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                epoch_loss += loss.item() * training_args.gradient_accumulation_steps
                
                grad_norm = None
                if (step + 1) % training_args.gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, training_args.max_grad_norm)
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                postfix = {"loss": f"{loss.item() * training_args.gradient_accumulation_steps:.4f}"}
                if grad_norm is not None:
                    postfix["grad_norm"] = f"{grad_norm.item():.4f}"
                progress_bar.set_postfix(postfix, refresh=False)
            
            avg_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

            run_eval_predict(epoch_idx=epoch + 1)
        
        # Save model (only trainable parameters for prefix tuning)
        output_dir = training_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save PEFT model
        if model_args.pre_seq_len is not None:
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Model saved to {output_dir}")

    if not training_args.do_train:
        run_eval_predict()

    logger.info("Done!")


if __name__ == "__main__":
    main()
