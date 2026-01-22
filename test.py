from transformers import AutoModel, AutoTokenizer

TOKENIZER_PATH = "/home/changc/chatglm2-6b"
MODEL_PATH = "/home/changc/chatglm2-6b"


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    print(f"Vocab size: {tokenizer.vocab_size}")

    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(model)


if __name__ == "__main__":
    main()
