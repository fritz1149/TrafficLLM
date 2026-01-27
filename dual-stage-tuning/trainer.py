# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
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
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""
import os
from typing import Optional
from transformers import Trainer
import torch
import math
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.utils import logging

logger = logging.get_logger(__name__)

WEIGHTS_NAME = "pytorch_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"


class PrefixTrainer(Trainer):
    def __init__(self, *args, save_changed=False, **kwargs):
        self.save_changed = save_changed
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        logging_steps = getattr(self.args, "logging_steps", 0) or 0
        if logging_steps > 0 and self.is_world_process_zero() and (self.state.global_step % logging_steps == 0):
            try:
                lr = None
                if getattr(self, "optimizer", None) is not None and len(self.optimizer.param_groups) > 0:
                    lr = self.optimizer.param_groups[0].get("lr", None)

                total_param_sq = 0.0
                total_grad_sq = 0.0
                prefix_param_sq = 0.0
                prefix_grad_sq = 0.0

                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if not p.requires_grad:
                            continue
                        param_sq = float(p.detach().float().pow(2).sum().item())
                        total_param_sq += param_sq
                        if "prompt_encoder" in name:
                            prefix_param_sq += param_sq

                        if p.grad is not None:
                            grad_sq = float(p.grad.detach().float().pow(2).sum().item())
                            total_grad_sq += grad_sq
                            if "prompt_encoder" in name:
                                prefix_grad_sq += grad_sq

                total_param_norm = math.sqrt(total_param_sq)
                total_grad_norm = math.sqrt(total_grad_sq)
                prefix_param_norm = math.sqrt(prefix_param_sq)
                prefix_grad_norm = math.sqrt(prefix_grad_sq)

                if lr is None:
                    logger.warning(
                        f"step={self.state.global_step} param_norm={total_param_norm:.6f} grad_norm={total_grad_norm:.6f} "
                        f"prefix_param_norm={prefix_param_norm:.6f} prefix_grad_norm={prefix_grad_norm:.6f}"
                    )
                else:
                    logger.warning(
                        f"step={self.state.global_step} lr={lr:.8g} param_norm={total_param_norm:.6f} grad_norm={total_grad_norm:.6f} "
                        f"prefix_param_norm={prefix_param_norm:.6f} prefix_grad_norm={prefix_grad_norm:.6f}"
                    )
            except Exception as e:
                logger.warning(f"Failed to compute norm stats: {e}")

        return loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if state_dict is None:
            state_dict = self.model.state_dict()
        if os.environ['MODEL'] != 'qwen' and (not isinstance(self.model, PreTrainedModel)):
            print("Trainer.model is not a `PreTrainedModel`")
            if self.save_changed:
                print("Saving Learnable Parameters")
                filtered_state_dict = {}
                for k, v in self.model.named_parameters():
                    if v.requires_grad:
                        filtered_state_dict[k] = state_dict[k]
                torch.save(filtered_state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            else:
                print("Saving the whole model 1")
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            if self.save_changed:
                print("Saving Learnable Parameters")
                filtered_state_dict = {}
                for k, v in self.model.named_parameters():
                    if v.requires_grad:
                        filtered_state_dict[k] = state_dict[k]
                self.model.save_pretrained(output_dir, state_dict=filtered_state_dict)
            else:
                print("Saving the whole model 2")
                self.model.save_pretrained(output_dir)
                # model.config.to_json_file(f"{output_dir}/adapter_config.json")
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
