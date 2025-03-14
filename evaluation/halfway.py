from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  confusion_matrix, classification_report
from tqdm import tqdm
import fire
import os
import torch
from transformers import AutoConfig
import sys
import json
import os
import numpy as np
import pickle
from pathlib import Path
from torchvision.models.feature_extraction import create_feature_extractor

def main(model_name,
         train_file: str = None,
         test_file: str = None,
         ptuning_path: str = None,
         transformer_index: int = None,
         output_path: str = None,
         **kwargs):

    if test_file is not None:
        assert os.path.exists(test_file), f"Provided Test file does not exist {test_file}"
        with open(test_file, "r", encoding="utf-8") as fin:
            test_set = fin.readlines()
    else:
        print("No Test file provided. Exiting.")
        sys.exit(1)

    if train_file is not None:
        assert os.path.exists(train_file), f"Provided train file does not exist {train_file}"
        with open(train_file, "r", encoding="utf-8") as fin:
            train_set = fin.readlines()
    else:
        print("No train file provided. Exiting.")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if ptuning_path is not None:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, pre_seq_len=128)
        model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        prefix_state_dict = torch.load(
            os.path.join(ptuning_path, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

        model = model.half()
        if os.environ["CPU"] == 1:
            model = model.to("cpu")
        else:
            model = model.cuda()
        model.transformer.prefix_encoder.float()

    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half()
    
    model = model.eval()
    # 钩子函数
    from_hook = None
    def hook(module, fea_in, fea_out):
        nonlocal from_hook
        from_hook = fea_out.data
        return None
    module_name = f"transformer.encoder.layers.{transformer_index}.mlp.dense_4h_to_h"
    for (name, module) in model.named_modules():
        if name == module_name:
            module.register_forward_hook(hook=hook)

    # 执行并输出整理后的数据
    def output(dataset, output_name):
        output = {
            "X": [],
            "y": []
        }
        for data in tqdm(dataset):
            promt = json.loads(data)["instruction"]
            label = json.loads(data)["output"]
            response, history = model.chat(tokenizer, promt, history=[], top_p=0.85, temperature=0.1)
            output["X"].append(from_hook.squeeze().cpu().tolist())
            output["y"].append(label)
        output["X"] = np.array(output["X"])
        print(output["X"].shape)

        os.makedirs(output_path, exist_ok=True)
        with open(Path(output_path, f"{output_name}.pkl"), "wb") as file:
            pickle.dump(output, file)
    
    output(test_set, "test")
    output(train_set, "train")

if __name__ == "__main__":
    fire.Fire(main)
