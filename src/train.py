import torch
from peft import LoraConfig, get_peft_model
from model import TransMambaLong
from dataset import get_dataloaders
from tqdm import tqdm

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Đang sử dụng thiết bị: {device}")

    base_model = TransMambaLong()
    lora_config = LoraConfig(
        r=8, lora_alpha=16, 
        target_modules=["query", "value"], 
        lora_dropout=0.1, bias="none"
    )
    model = get_peft_model(base_model, lora_config).to(device)
    model.print_trainable_parameters()

    train_loader, _ = get_dataloaders(batch_size=2, max_length=1024)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()

    EPOCHS = 1 
    print(f"\n[*] BẮT ĐẦU HUẤN LUYỆN ({EPOCHS} EPOCHS) ====")
    
    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(ids, mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "VRAM": f"{torch.cuda.memory_allocated()/1e9:.1f}GB"})

    save_path = "transmamba_long_weights.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n[+] ĐÃ LƯU TRỌNG SỐ MÔ HÌNH TẠI: {save_path}")

if __name__ == "__main__":
    main()