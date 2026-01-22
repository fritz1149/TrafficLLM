import pickle
import fire
import json
from preprocess_utils import (
    build_td_text_dataset,
    build_tg_text_dataset,
    build_tu_text_dataset,
    write_labels,
    build_dataset,
    write_dataset
)

def main(src_dir, output_name, standard, granularity):
    with open(src_dir, 'rb') as file:
        data = pickle.load(file)
    
    label_map_origin = data['label_map']
    label_map = {}
    for key, value in label_map_origin.items():
        label_map[value] = key

    with open('instructions.json', 'r') as file:
        instruction = json.load(file)[standard]

    dataset = []
    for unit in data['data']:
        dataset.append(
            {
                "instruction": instruction.format(granularity, granularity, unit[0]),
                "output": label_map[unit[1]]
            }
        )

    import random
    train_rate = 0.9
    all_indices = [x for x in range(len(dataset))]
    train_num = round(len(dataset) * train_rate)
    train_indices = random.sample(all_indices, train_num)
    train_dataset = [dataset[x] for x in train_indices]
    test_dataset = [dataset[x] for x in all_indices if x not in train_indices]
    write_dataset(train_dataset, f'../datasets/{output_name}/{output_name}_detection_{granularity}_train.json')
    write_dataset(test_dataset, f'../datasets/{output_name}/{output_name}_detection_{granularity}_test.json')

    labels = list(label_map.keys())
    write_labels(labels, f'../datasets/{output_name}/{output_name}_label.json')
    
if __name__ == '__main__':
    fire.Fire(main)