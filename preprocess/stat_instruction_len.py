import argparse
import json
import math
import os
from collections import Counter

from transformers import AutoTokenizer


DEFAULT_SUFFIXES = (".json", ".jsonl")


def iter_data_files(data_dir, suffixes=DEFAULT_SUFFIXES):
    if os.path.isfile(data_dir):
        yield data_dir
        return

    for name in sorted(os.listdir(data_dir)):
        file_path = os.path.join(data_dir, name)
        if not os.path.isfile(file_path):
            continue
        if suffixes is None or name.endswith(suffixes):
            yield file_path


def iter_instructions(file_path):
    with open(file_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "instruction" not in record:
                continue
            instruction = record["instruction"]
            if not isinstance(instruction, str):
                instruction = str(instruction)
            yield instruction


def percentile_from_counts(length_counts, percentile):
    total = sum(length_counts.values())
    if total == 0:
        return None
    target = math.ceil(total * percentile / 100)
    cumulative = 0
    for length, count in sorted(length_counts.items()):
        cumulative += count
        if cumulative >= target:
            return length
    return None


def main():
    parser = argparse.ArgumentParser(description="统计 instruction 的 input_ids 长度分布")
    parser.add_argument("--tokenizer_path", required=True, help="tokenizer 目录")
    parser.add_argument("--data_dir", required=True, help="预处理数据目录或文件")
    parser.add_argument(
        "--suffixes",
        nargs="*",
        default=list(DEFAULT_SUFFIXES),
        help="需要统计的文件后缀，例如 .json .jsonl",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="最多统计的样本数")
    parser.add_argument("--output", default=None, help="保存长度分布到 json 文件")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    length_counts = Counter()
    total = 0
    suffixes = tuple(args.suffixes) if args.suffixes else None

    for file_path in iter_data_files(args.data_dir, suffixes=suffixes):
        for instruction in iter_instructions(file_path):
            input_ids = tokenizer(
                instruction,
                add_special_tokens=True,
                return_attention_mask=False,
                return_token_type_ids=False,
            ).input_ids
            length_counts[len(input_ids)] += 1
            total += 1
            if args.max_samples is not None and total >= args.max_samples:
                break
        if args.max_samples is not None and total >= args.max_samples:
            break

    if total == 0:
        print("未找到可统计的 instruction")
        return

    min_len = min(length_counts.keys())
    max_len = max(length_counts.keys())
    mean_len = sum(length * count for length, count in length_counts.items()) / total

    p = {}
    for pct in range(50, 100, 5):
        p[pct] = percentile_from_counts(length_counts, pct)

    print(f"Total samples: {total}")
    print(f"Min length: {min_len}")
    print(f"Max length: {max_len}")
    print(f"Mean length: {mean_len:.2f}")
    percentile_str = "\n".join(f"p{pct}={val}" for pct, val in sorted(p.items()))
    print(f"Percentiles: {percentile_str}")
    print("Length distribution (length\tcount):")
    for length, count in sorted(length_counts.items()):
        print(f"{length}\t{count}")

    if args.output:
        output_payload = {
            "total": total,
            "min": min_len,
            "max": max_len,
            "mean": mean_len,
            "percentiles": {f"p{pct}": val for pct, val in p.items()},
            "distribution": dict(sorted(length_counts.items())),
        }
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fout:
            json.dump(output_payload, fout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
