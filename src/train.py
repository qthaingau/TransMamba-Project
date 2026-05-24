import torch
import time
from peft import LoraConfig, get_peft_model
from model import TransMambaLong
from dataset import get_dataloaders
from tqdm import tqdm

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Đang sử dụng thiết bị: {device}")

    # Khởi tạo mô hình và áp dụng LoRA
    base_model = TransMambaLong()
    lora_config = LoraConfig(
        r=8, lora_alpha=16, 
        target_modules=["query", "value"], 
        lora_dropout=0.1, bias="none"
    )
    model = get_peft_model(base_model, lora_config).to(device)
    model.print_trainable_parameters()

    # Cấu hình siêu tham số (Strict Baseline Comparison)
    train_loader, _ = get_dataloaders(batch_size=2, max_length=1024)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()

    EPOCHS = 1 
    print(f"\n[*] BẮT ĐẦU HUẤN LUYỆN BẢN 4 LỚP ({EPOCHS} EPOCH) ====")
    
    start_time = time.time()
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

    train_time = (time.time() - start_time) / 60
    print(f"\n[+] Hoàn thành huấn luyện trong: {train_time:.2f} phút")

    save_path = "transmamba_long_4layers.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[+] ĐÃ LƯU TRỌNG SỐ MÔ HÌNH TẠI: {save_path}")

if __name__ == "__main__":
    main()