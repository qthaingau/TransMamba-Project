import torch
import torch.nn as nn
from transformers import AutoModel
from mamba_ssm import Mamba

class TransMambaLong(nn.Module):
    def __init__(self, encoder_name="allenai/longformer-base-4096", d_model=768, num_classes=2):
        super(TransMambaLong, self).__init__()
        
        # Encoder: Longformer (Global Context)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        
        # Decoder: Mamba Layers (Linear Sequential Context)
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) 
            for _ in range(8)
        ])
        self.mamba_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(8)])

        # Fusion: Cross-Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        self.final_norm = nn.LayerNorm(d_model)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        # 1. Longformer Encoder
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # 2. Mamba Decoder Stack
        h = enc_out
        for mamba, norm in zip(self.mamba_layers, self.mamba_norms):
            h = mamba(norm(h)) + h
            
        # 3. Cross-Attention Fusion (Query from Mamba, Key/Value from Longformer)
        fused, _ = self.cross_attn(query=h, key=enc_out, value=enc_out)
        
        # 4. Final Output
        out = self.final_norm(h + fused)
        pooled = out.mean(dim=1) # Global Average Pooling
        return self.classifier(pooled)

from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=4, max_length=1024):
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    ds = load_dataset("imdb")

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)

    tokenized_ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    tokenized_ds.set_format("torch")

    train_loader = DataLoader(tokenized_ds["train"], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(tokenized_ds["test"], batch_size=batch_size)
    
    return train_loader, test_loader
import torch
from peft import LoraConfig, get_peft_model
from model import TransMambaLong
from dataset import get_dataloaders
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Init Model & PEFT
base_model = TransMambaLong()
lora_config = LoraConfig(
    r=8, lora_alpha=16, 
    target_modules=["query", "value"], # Target Longformer Attention
    lora_dropout=0.1, bias="none"
)
model = get_peft_model(base_model, lora_config).to(device)

# 2. Data & Optimizer
train_loader, _ = get_dataloaders(batch_size=2) # Small batch for 1024 tokens
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

# 3. Training Loop
model.train()
for epoch in range(3):
    pbar = tqdm(train_loader)
    for batch in pbar:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        pbar.set_description(f"Loss: {loss.item():.4f} | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# Save to Kaggle Working Directory
torch.save(model.state_dict(), "/kaggle/working/transmamba_long.pth")
