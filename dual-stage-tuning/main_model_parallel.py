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
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
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


def setup_ddp():
    """Initialize DDP environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_ddp():
    """Clean up DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(use_distributed):
    """Check if current process is the main process."""
    if not use_distributed:
        return True
    return dist.get_rank() == 0


def td_evaluation(predict_responses, target_responses, label_file):
    """
    Evaluation logic from evaluation.py - compute macro metrics.
    For predictions not in label_dict: the true class is counted as FN,
    all other classes are counted as TN. The unknown prediction is NOT
    treated as a separate class in macro averaging.
    Note: every sample's ground truth is guaranteed to be in label_dict.
    """
    with open(label_file, "r", encoding="utf-8") as fin:
        label_dict = json.load(fin)

    num_classes = len(label_dict)
    # per-class accumulators: index by label_id
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    correct = 0
    total = len(predict_responses)
    invalid_count = 0

    for predict_response, target_response in zip(predict_responses, target_responses):
        if ' ' in predict_response:
            predict_response = predict_response.split(" ")[-1]
        label_id = label_dict[target_response]  # always valid

        if predict_response in label_dict:
            pred_id = label_dict[predict_response]
            if pred_id == label_id:
                tp[label_id] += 1
                correct += 1
            else:
                fp[pred_id] += 1
                fn[label_id] += 1
        else:
            # Unknown prediction: true class is FN, all others are TN
            fn[label_id] += 1
            invalid_count += 1

    # Per-class precision / recall / f1
    per_precision = []
    per_recall = []
    per_f1 = []
    for c in range(num_classes):
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_precision.append(p)
        per_recall.append(r)
        per_f1.append(f)

    macro_precision = sum(per_precision) / num_classes if num_classes > 0 else 0.0
    macro_recall = sum(per_recall) / num_classes if num_classes > 0 else 0.0
    macro_f1 = sum(per_f1) / num_classes if num_classes > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "f1_macro": macro_f1,
    }

    # Build label name lookup for report
    id_to_label = {v: k for k, v in label_dict.items()}
    print("acc:", metrics["accuracy"])
    print("precision:", metrics["precision_macro"])
    print("recall:", metrics["recall_macro"])
    print("f1:", metrics["f1_macro"])
    print("per-class metrics:")
    for c in range(num_classes):
        print(f"  {id_to_label.get(c, c)}: precision={per_precision[c]:.4f}  recall={per_recall[c]:.4f}  f1={per_f1[c]:.4f}  tp={tp[c]}  fp={fp[c]}  fn={fn[c]}")
    print(f"invalid predictions: {invalid_count} / {total}")

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
        "base_model.model.model.language_model.embed_tokens": "cuda:0",
        "base_model.model.model.language_model.rotary_emb": "cuda:0",
        "base_model.model.model.visual": "cuda:0",
        "word_embeddings": "cuda:0",
        "prompt_encoder": "cuda:0",
        "base_model.model.lm_head": "cuda:1",
        "base_model.model.model.language_model.norm": "cuda:1",
        **{f"base_model.model.model.language_model.layers.{i}": "cuda:0" for i in range(0, split_layer)},
        **{f"base_model.model.model.language_model.layers.{i}": "cuda:1" for i in range(split_layer, num_hidden_layers)},
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
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        for i in range(outputs.shape[0]):
            generated_ids = outputs[i][input_ids.shape[1]:]
            pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            pred_text = pred_text.replace("<|im_end|>", "").strip()
            predictions.append(pred_text)
    
    return predictions, targets


def single_input_test(model, tokenizer, data_args, max_new_tokens, input_text):
    """
    对单条用户输入文本执行一次生成测试。
    使用与 preprocess_function_eval 相同的数据处理逻辑。
    """
    QWENVL_PAD_TOKEN = tokenizer.special_tokens_map.get("pad_token", None)
    assert QWENVL_PAD_TOKEN is not None
    QWENVL_PAD_ID = tokenizer.convert_tokens_to_ids([QWENVL_PAD_TOKEN])[0]

    query = f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(query)
    origin_len = len(input_ids)
    pad_len = data_args.max_source_length - origin_len
    if pad_len < 0:
        print(f"[WARNING] input length {origin_len} exceeds max_source_length {data_args.max_source_length}, truncating.")
        input_ids = input_ids[-data_args.max_source_length:]
        pad_len = 0
        origin_len = len(input_ids)
    input_ids_padded = [QWENVL_PAD_ID] * pad_len + input_ids
    attention_mask = [0] * pad_len + [1] * origin_len

    input_ids_tensor = torch.tensor([input_ids_padded], dtype=torch.long).to("cuda:0")
    attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long).to("cuda:0")

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    generated_ids = outputs[0][input_ids_tensor.shape[1]:]
    pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    pred_text = pred_text.replace("<|im_end|>", "").strip()

    print("=" * 60)
    print("[single_input_test] Input:")
    print(input_text)
    print("[single_input_test] Output:")
    print(pred_text)
    print("=" * 60)
    return pred_text


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

    # DDP/FSDP setup
    use_ddp = getattr(model_args, "use_ddp", False)
    use_fsdp = getattr(model_args, "use_fsdp", False)
    local_rank, global_rank, world_size = 0, 0, 1
    if use_ddp or use_fsdp:
        local_rank, global_rank, world_size = setup_ddp()
        logger.info(f"Distributed initialized: local_rank={local_rank}, global_rank={global_rank}, world_size={world_size}")

    # Check GPU availability
    n_gpus = torch.cuda.device_count()
    
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
    if training_args.bf16:
        qwen_vl_dtype = torch.bfloat16
    elif training_args.fp16:
        qwen_vl_dtype = torch.float16
    else:
        qwen_vl_dtype = torch.float32

    # Load Qwen-VL model
    from transformers import Qwen3VLForConditionalGeneration
    
    if model_args.flash_attn:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            dtype=qwen_vl_dtype,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )
        print("flash attn enabled")
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            dtype=qwen_vl_dtype,
            low_cpu_mem_usage=True,
        )
    
    # Replace visual encoder with Identity
    model.model.visual = nn.Identity()
    
    # Apply PEFT prefix tuning
    
    from peft import PrefixTuningConfig, LoraConfig, get_peft_model, TaskType
    if getattr(model_args, "peft_type", "prefix") == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, peft_config)
    else:
        peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=model_args.pre_seq_len, prefix_projection=model_args.prefix_projection)
        model = get_peft_model(model, peft_config)
        prompt_encoder_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
        model.prompt_encoder = model.prompt_encoder.to(dtype=prompt_encoder_dtype)

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")
    model.print_trainable_parameters()

    # Get number of hidden layers
    num_hidden_layers = 36
    
    # Create device map and dispatch model
    if getattr(model_args, "model_parallel", False):
        device_map, split_layer = create_device_map(model_args, num_hidden_layers)
        print(f"Using model parallel with split_layer={split_layer}, num_hidden_layers={num_hidden_layers}")
        print(f"Device map: {device_map}")
        
        model = dispatch_model(model, device_map=device_map)
        # Patch get_prompt for model parallel
        patch_get_prompt_for_mp(model, split_layer)
    elif use_fsdp:
        # FSDP mode: shards model across GPUs for memory efficiency
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer
        
        # Enable gradient checkpointing BEFORE FSDP wrapping for LoRA
        is_lora = getattr(model_args, "peft_type", "prefix") == "lora"
        if is_lora:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            logger.info("Gradient checkpointing enabled before FSDP wrapping")
        
        # Convert all parameters to uniform dtype before FSDP wrapping
        # This is required because LoRA adapters are created in float32 by default
        target_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
        model = model.to(target_dtype)
        logger.info(f"Converted all model parameters to {target_dtype} for FSDP")
        
        # Mixed precision policy
        mp_policy = MixedPrecision(
            param_dtype=target_dtype,
            reduce_dtype=target_dtype,
            buffer_dtype=target_dtype,
        )
        
        # Auto wrap policy for transformer layers
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen3VLTextDecoderLayer},
        )
        
        torch.cuda.set_device(local_rank)
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            auto_wrap_policy=auto_wrap_policy,
            # cpu_offload=CPUOffload(offload_params=True),  # Offload params to CPU to save GPU memory
            device_id=local_rank,
            use_orig_params=True,  # Required for LoRA
        )
        logger.info(f"Model wrapped with FSDP on cuda:{local_rank} with CPU offload")
    elif use_ddp:
        # DDP mode: place model on local GPU
        model = model.to(f"cuda:{local_rank}")
        # static_graph=True is required for compatibility with gradient_checkpointing
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, static_graph=True)
        logger.info(f"Model wrapped with DDP on cuda:{local_rank}")
    else:
        # Single GPU mode: move model to cuda:0
        model = model.to("cuda:0")
        logger.info("Model moved to cuda:0 (single GPU mode)")

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
        
        # Use DistributedSampler for DDP/FSDP
        use_distributed = use_ddp or use_fsdp
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True) if use_distributed else None
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=0,
        )
        
        # Setup optimizer - only optimize trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        # bf16 + fp32 master weights for better numerical stability
        use_master_weights = getattr(model_args, "bf16_master_weights", False) and training_args.bf16
        if use_master_weights:
            # Create fp32 copies of trainable parameters as master weights
            master_params = [p.detach().clone().float().requires_grad_(True) for p in trainable_params]
            optimizer = AdamW(master_params, lr=training_args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
            logger.info("Using bf16 training with fp32 master weights for better numerical stability")
        else:
            master_params = None
            optimizer = AdamW(trainable_params, lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
        
        # Setup scheduler
        num_training_steps = len(train_dataloader) * int(training_args.num_train_epochs) // training_args.gradient_accumulation_steps
        num_warmup_steps = training_args.warmup_steps if training_args.warmup_steps > 0 else int(num_training_steps * training_args.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        
        num_epochs = int(training_args.num_train_epochs)
        if num_epochs > data_args.stop_epochs:
            num_epochs = data_args.stop_epochs
        global_step = 0
        
        # Mixed precision scaler (not needed for FSDP - it handles mixed precision internally)
        # Also GradScaler is only for fp16, not bf16
        scaler = torch.amp.GradScaler('cuda') if (training_args.fp16 and not use_fsdp) else None
        use_amp = (training_args.fp16 or training_args.bf16) and not use_fsdp  # FSDP handles AMP via MixedPrecision
        amp_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
        
        # Enable gradient checkpointing for LoRA mode (skip if FSDP, already enabled before wrapping)
        is_lora = getattr(model_args, "peft_type", "prefix") == "lora"
        if is_lora and not use_fsdp:
            base_model = model.module if use_ddp else model
            base_model.gradient_checkpointing_enable()
            base_model.enable_input_require_grads()
            logger.info("Gradient checkpointing enabled for LoRA training")
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()
            accumulated_steps = 0
            
            # Set epoch for DistributedSampler to ensure proper shuffling
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not is_main_process(use_distributed))
            
            for step, batch in enumerate(progress_bar):
                # Move to appropriate GPU based on mode
                if use_ddp or use_fsdp:
                    device = f"cuda:{local_rank}"
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                elif getattr(model_args, "model_parallel", False):
                    # Model parallel: dispatch_model will handle cross-GPU
                    input_ids = batch["input_ids"].to("cuda:0")
                    attention_mask = batch["attention_mask"].to("cuda:0")
                    labels = batch["labels"].to("cuda:1")
                else:
                    # Single GPU mode
                    input_ids = batch["input_ids"].to("cuda:0")
                    attention_mask = batch["attention_mask"].to("cuda:0")
                    labels = batch["labels"].to("cuda:0")
                
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
                
                if not torch.isfinite(loss):
                    postfix = {"loss": f"{loss.item() * training_args.gradient_accumulation_steps:.4f}"}
                    progress_bar.set_postfix(postfix, refresh=False)
                    sys.stdout.flush()
                    sys.stderr.flush()
                    continue
                
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                epoch_loss += loss.item() * training_args.gradient_accumulation_steps
                accumulated_steps += 1
                
                grad_norm = None
                if accumulated_steps == training_args.gradient_accumulation_steps:
                    # For master weights: copy gradients from bf16 params to fp32 master params
                    if use_master_weights:
                        for mp, p in zip(master_params, trainable_params):
                            if p.grad is not None:
                                if mp.grad is None:
                                    mp.grad = p.grad.float()
                                else:
                                    mp.grad.copy_(p.grad)
                    
                    if scaler:
                        scaler.unscale_(optimizer)
                    
                    # Check gradient finiteness
                    params_to_check = master_params if use_master_weights else trainable_params
                    grads_finite = True
                    for p in params_to_check:
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            grads_finite = False
                            break
                    if grads_finite:
                        grad_norm = torch.nn.utils.clip_grad_norm_(params_to_check, training_args.max_grad_norm)
                    else:
                        grad_norm = torch.tensor(float("nan"))
                    if scaler:
                        old_scale = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        new_scale = scaler.get_scale()
                        if new_scale >= old_scale:
                            # For master weights: copy updated fp32 weights back to bf16 model params
                            if use_master_weights:
                                for mp, p in zip(master_params, trainable_params):
                                    p.data.copy_(mp.data)
                            scheduler.step()
                            global_step += 1
                    else:
                        if not torch.isfinite(grad_norm):
                            optimizer.zero_grad()
                            accumulated_steps = 0
                            postfix = {"loss": f"{loss.item() * training_args.gradient_accumulation_steps:.4f}"}
                            postfix["grad_norm"] = f"{grad_norm.item():.4f}"
                            progress_bar.set_postfix(postfix, refresh=False)
                            sys.stdout.flush()
                            sys.stderr.flush()
                            continue
                        optimizer.step()
                        
                        # For master weights: copy updated fp32 weights back to bf16 model params
                        if use_master_weights:
                            for mp, p in zip(master_params, trainable_params):
                                p.data.copy_(mp.data)
                        scheduler.step()
                        global_step += 1
                    optimizer.zero_grad()
                    accumulated_steps = 0
                
                postfix = {"loss": f"{loss.item() * training_args.gradient_accumulation_steps:.4f}"}
                if grad_norm is not None:
                    postfix["grad_norm"] = f"{grad_norm.item():.4f}"
                progress_bar.set_postfix(postfix, refresh=False)
                
                sys.stdout.flush()
                sys.stderr.flush()
            
            avg_loss = epoch_loss / len(train_dataloader)
            if is_main_process(use_distributed):
                logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

            # Disable gradient checkpointing before eval/predict for LoRA mode (skip for FSDP)
            if is_lora and not use_fsdp:
                base_model.gradient_checkpointing_disable()
                base_model.disable_input_require_grads()
            
            # Synchronize before eval/predict
            if use_distributed:
                dist.barrier()
            
            # Only run eval/predict on main process
            if is_main_process(use_distributed) and (epoch == 3 or epoch == 4):
                run_eval_predict(epoch_idx=epoch + 1)
            
            # Synchronize after eval/predict
            if use_distributed:
                dist.barrier()
            
            # Re-enable gradient checkpointing after eval/predict for LoRA mode (skip for FSDP)
            if is_lora and not use_fsdp:
                base_model.gradient_checkpointing_enable()
                base_model.enable_input_require_grads()
            
            sys.stdout.flush()
            sys.stderr.flush()
        
        # Save model (only trainable parameters for prefix tuning) - only on main process
        # if is_main_process(use_ddp):
        #     output_dir = training_args.output_dir
        #     os.makedirs(output_dir, exist_ok=True)
            
        #     # Save PEFT model
        #     save_model = model.module if use_ddp else model
        #     if model_args.pre_seq_len is not None or getattr(model_args, "peft_type", "prefix") == "lora":
        #         save_model.save_pretrained(output_dir)
        #         tokenizer.save_pretrained(output_dir)
        #         logger.info(f"Model saved to {output_dir}")

    if getattr(data_args, "single_input_text", None) is not None:
        logger.info("Running single input test...")
        single_input_test(
            model=model,
            tokenizer=tokenizer,
            data_args=data_args,
            max_new_tokens=data_args.max_target_length,
            input_text=data_args.single_input_text,
        )
    elif not training_args.do_train:
        use_distributed = use_ddp or use_fsdp
        if is_main_process(use_distributed):
            run_eval_predict()

    # Cleanup distributed
    if use_ddp or use_fsdp:
        cleanup_ddp()
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
