from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from transformers.modeling_utils import PreTrainedModel
import fire
import torch
import json
import os
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_model(model, ptuning_path):
    if ptuning_path is not None:
        prefix_state_dict = torch.load(
            os.path.join(ptuning_path, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

        model = model.half().cuda()
        model.transformer.prefix_encoder.float()

    return model

prompt_style = """### Instruction:
You are an expert with advanced knowledge in understanding network traffic and distinguishing different kinds of traffic.
Please answer the following network traffic identification  question.

### Question:
{}

### Response:
{}
"""

def main(model_path, text, ptuning_path, ptuning=False):

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if 'qwen' in model_path:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config)
        if ptuning:    
            from peft import PeftModel
            print(f"ptuning_path: {ptuning_path }")
            model = PeftModel.from_pretrained(model, ptuning_path)
            # model = model.half()
            # for k, v in model.named_parameters():
            #     if v.requires_grad:
            #         v.float()
    else:
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=128)
        model = AutoModel.from_pretrained(model_path, config=model_config, trust_remote_code=True)
        model_downstream = load_model(model, downstream_model_path)
        
    model = model.eval()
    model = model.to("cuda")

    if 'qwen' in model_path:
        text = prompt_style.format(text, "")
        print(f"input: {text}")
        device="cuda"
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        outputs = model.generate(
            input_ids=model_inputs.input_ids,
            max_new_tokens=2048,
            attention_mask= model_inputs.attention_mask
        )
        response = tokenizer.batch_decode(outputs)[0]
    else:
        response, history = model.chat(tokenizer, text, history=[])
    print(f"response: {response}")


if __name__ == "__main__":
    fire.Fire(main)
