import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=4, max_length=1024):
    print("[*] Đang tải Tokenizer và Dataset IMDB...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    ds = load_dataset("imdb")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    print("[*] Đang Tokenize dữ liệu...")
    tokenized_ds = ds.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_ds.set_format("torch")

    train_loader = DataLoader(tokenized_ds["train"], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(tokenized_ds["test"], batch_size=batch_size)
    
    print(f"[+] Hoàn tất! Kích thước Batch Train: {len(train_loader)}")
    return train_loader, test_loader