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

def main(model_name,
         test_file: str = None,
         label_file: str = None,
         traffic_task: str = None,
         ptuning_path: str = None,
         **kwargs):

    if test_file is not None:
        assert os.path.exists(test_file), f"Provided Test file does not exist {test_file}"
        with open(test_file, "r", encoding="utf-8") as fin:
            test_set = fin.readlines()
            # test_set = json.load(fin)
    else:
        print("No Test file provided. Exiting.")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if ptuning_path is not None:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, pre_seq_len=128)
        model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)

    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
    
    print(model)


if __name__ == "__main__":
    fire.Fire(main)
