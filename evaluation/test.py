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
from peft import PrefixTuningConfig, get_peft_model, TaskType

def main(model_name,
         ptuning_path,
         **kwargs):

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(type(tokenizer), tokenizer.eos_token_id, tokenizer.pad_token_id)

if __name__ == "__main__":
    fire.Fire(main)
