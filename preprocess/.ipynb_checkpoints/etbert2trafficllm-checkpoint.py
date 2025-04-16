import pickle
import fire
import json
from preprocess_utils import (
    build_td_text_dataset,
    build_tg_text_dataset,
    build_tu_text_dataset,
    write_labels,
    build_dataset,
    save_dataset
)

def main(src_dir, dst_dir, output_name):
    with open(src_dir, 'rb') as file:
        data = pickle.load(src_dir)
    
        

if __name__ == '__main__':
    fire.Fire(main)