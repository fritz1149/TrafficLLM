from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  confusion_matrix, classification_report
from tqdm import tqdm
import fire
import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM
import sys
import json
import os
import numpy as np
import pickle
from pathlib import Path
from peft import PrefixTuningConfig, get_peft_model, TaskType

def main(model_name,
         **kwargs):

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    print(model)

if __name__ == "__main__":
    fire.Fire(main)
