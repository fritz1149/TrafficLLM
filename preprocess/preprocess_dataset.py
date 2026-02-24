from specfic_dataset_utils import ustc_tfc2016_preprocess
from preprocess_utils import (
    build_td_text_dataset,
    build_tg_text_dataset,
    build_tu_text_dataset,
    get_td_prompt_prefix_suffix,
    write_labels,
    build_dataset,
    build_dataset_from_split,
    save_dataset
)
from tqdm import tqdm
import argparse
import random
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="raw dataset path", required=True)
    parser.add_argument("--dataset_name", type=str, help="dataset name", required=True)
    parser.add_argument("--traffic_task", type=str, help="traffic task", required=True, choices=["detection", "generation", "understanding"])
    parser.add_argument("--granularity", type=str, help="processing granularity", required=True, choices=["flow", "packet"])
    parser.add_argument("--output_path", type=str, help="output dataset path", required=True)
    parser.add_argument("--output_name", type=str, help="output dataset name", required=True)
    parser.add_argument("--sampling_method", type=str, help="sampling_method", required=False)
    parser.add_argument("--max_sampling_number", type=int, help="max_sampling_number", required=False)
    parser.add_argument("--split_mode", action="store_true", help="load train/test/val splits from directories")
    parser.add_argument("--inference_mode", action="store_true")
    parser.add_argument("--flow_feature", type=str, help="flow feature type", required=True)
    parser.add_argument("--max_packet_number", type=int, help="max_packet_number", required=False)
    parser.add_argument("--max_token_length", type=int, help="max_token_length", required=False)
    parser.add_argument("--tokenizer_path", type=str, help="tokenizer_path", required=False)

    args = parser.parse_args()
    return args


def _list_label_dirs(root_dir):
    return [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]


def traffic_detection_preprocess(args, detection_task):
    """Dataset preprocessing for the traffic detection (TD) task"""
    train_dataset = []
    test_dataset = []
    labels = []

    args.prompt_prefix, args.prompt_suffix = get_td_prompt_prefix_suffix(
        detection_task,
        args.granularity,
    )

    if args.split_mode:
        train_root = os.path.join(args.input, "train")
        test_root = os.path.join(args.input, "test")
        val_root = os.path.join(args.input, "val")

        files = _list_label_dirs(train_root)
        labels.extend(files)
        val_dataset = []

        for file in tqdm(files):
            train_data = build_dataset_from_split(args, train_root, file)
            test_data = build_dataset_from_split(args, test_root, file)
            val_data = build_dataset_from_split(args, val_root, file)

            train_text_data = build_td_text_dataset(train_data, second_label=file, task_name=detection_task, granularity=args.granularity)
            test_text_data = build_td_text_dataset(test_data, second_label=file, task_name=detection_task, granularity=args.granularity)
            val_text_data = build_td_text_dataset(val_data, second_label=file, task_name=detection_task, granularity=args.granularity)

            train_dataset.extend(train_text_data)
            test_dataset.extend(test_text_data)
            val_dataset.extend(val_text_data)

        save_dataset(args, train_dataset, test_dataset, val_dataset=val_dataset)
    else:
        files = os.listdir(args.input)
        labels.extend(files)

        for file in tqdm(files):
            train_data, test_data = build_dataset(args, args.input, file, sampling_method=args.sampling_method)

            train_text_data = build_td_text_dataset(train_data, second_label=file, task_name=detection_task, granularity=args.granularity)
            test_text_data = build_td_text_dataset(test_data, second_label=file, task_name=detection_task, granularity=args.granularity)

            train_dataset.extend(train_text_data)
            test_dataset.extend(test_text_data)

        save_dataset(args, train_dataset, test_dataset)

    write_labels(labels, os.path.join(args.output_path, args.output_name + "_label.json"))


def traffic_generation_preprocess(args):
    """Dataset preprocessing for the traffic generation (TG) task"""

    train_dataset = []
    test_dataset = []
    labels = []

    if args.split_mode:
        train_root = os.path.join(args.input, "train")
        test_root = os.path.join(args.input, "test")
        val_root = os.path.join(args.input, "val")

        files = _list_label_dirs(train_root)
        labels.extend(files)
        val_dataset = []

        for file in tqdm(files):
            train_data = build_dataset_from_split(args, train_root, file)
            test_data = build_dataset_from_split(args, test_root, file)
            val_data = build_dataset_from_split(args, val_root, file)

            train_text_data = build_tg_text_dataset(train_data, traffic_label=file, granularity=args.granularity)
            test_text_data = build_tg_text_dataset(test_data, traffic_label=file, granularity=args.granularity)
            val_text_data = build_tg_text_dataset(val_data, traffic_label=file, granularity=args.granularity)

            train_dataset.extend(train_text_data)
            test_dataset.extend(test_text_data)
            val_dataset.extend(val_text_data)

        save_dataset(args, train_dataset, test_dataset, val_dataset=val_dataset)
    else:
        files = os.listdir(args.input)
        labels.extend(files)

        for file in tqdm(files):
            train_data, test_data = build_dataset(args, args.input, file)

            train_text_data = build_tg_text_dataset(train_data, traffic_label=file, granularity=args.granularity)
            test_text_data = build_tg_text_dataset(test_data, traffic_label=file, granularity=args.granularity)

            train_dataset.extend(train_text_data)
            test_dataset.extend(test_text_data)

        save_dataset(args, train_dataset, test_dataset)


def traffic_understanding_preprocess(args):
    """Dataset preprocessing for the traffic understanding (TU) task"""
    train_dataset = []
    test_dataset = []
    labels = []

    args.granularity = "packet"
    if args.split_mode:
        train_root = os.path.join(args.input, "train")
        test_root = os.path.join(args.input, "test")
        val_root = os.path.join(args.input, "val")

        files = _list_label_dirs(train_root)
        labels.extend(files)
        val_dataset = []

        for file in tqdm(files):
            train_data = build_dataset_from_split(args, train_root, file)
            test_data = build_dataset_from_split(args, test_root, file)
            val_data = build_dataset_from_split(args, val_root, file)

            train_text_data = build_tu_text_dataset(train_data, fields=["TCP"])
            test_text_data = build_tu_text_dataset(test_data, fields=["TCP"])
            val_text_data = build_tu_text_dataset(val_data, fields=["TCP"])

            train_dataset.extend(train_text_data)
            test_dataset.extend(test_text_data)
            val_dataset.extend(val_text_data)

        save_dataset(args, train_dataset, test_dataset, val_dataset=val_dataset)
    else:
        files = os.listdir(args.input)
        labels.extend(files)

        for file in tqdm(files):
            train_data, test_data = build_dataset(args, args.input, file)

            train_text_data = build_tu_text_dataset(train_data, fields=["TCP"])
            test_text_data = build_tu_text_dataset(test_data, fields=["TCP"])

            train_dataset.extend(train_text_data)
            test_dataset.extend(test_text_data)

        random.shuffle(train_dataset)
        random.shuffle(test_dataset)
        train_dataset = train_dataset[:20000]
        test_dataset = test_dataset[:200]

        save_dataset(args, train_dataset, test_dataset)


def main():
    args = get_args()
    traffic_task = args.traffic_task

    if traffic_task == "detection":
        if args.dataset_name == "qq" or args.dataset_name == "weixin": #其一
            traffic_detection_preprocess(args, detection_task="MBD")
        elif args.dataset_name == "USTC-TFC": #其二
            # ustc_tfc2016_preprocess(args, detection_task="EMD")
            traffic_detection_preprocess(args, detection_task="EMD")
        elif args.dataset_name == "iscx-botnet":
            traffic_detection_preprocess(args, detection_task="BND")
        elif args.dataset_name == "ISCX-VPN-app" or args.dataset_name == "lfett-2021": #其三
            traffic_detection_preprocess(args, detection_task="EVD")
        elif args.dataset_name == "ISCX-VPN-service": #其四
            traffic_detection_preprocess(args, detection_task="EVD2")
        elif args.dataset_name == "dohbrw-2020":
            traffic_detection_preprocess(args, detection_task="MDD") #待做
        elif args.dataset_name == "iscx-tor-2016":
            traffic_detection_preprocess(args, detection_task="TBD")
        # elif args.dataset_name == "cic-adware":
        #     traffic_detection_preprocess(args, detection_task="ATD")
        # elif args.dataset_name == "cic-ransomware":
        #     traffic_detection_preprocess(args, detection_task="RTD")
        # elif args.dataset_name == "cic-scareware":
        #     traffic_detection_preprocess(args, detection_task="STD")
        elif args.dataset_name == "dapt-2020":
            traffic_detection_preprocess(args, detection_task="APT")
        elif args.dataset_name == "CSTNET":
            traffic_detection_preprocess(args, detection_task="EAC2")
        else:
            traffic_detection_preprocess(args, detection_task="EAC")

    elif traffic_task == "generation":
        traffic_generation_preprocess(args)

    else:
        traffic_understanding_preprocess(args)


if __name__ == "__main__":
    main()
