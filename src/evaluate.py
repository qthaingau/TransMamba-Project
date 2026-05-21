import torch
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, classification_report
from model import TransMambaLong
from dataset import get_dataloaders

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Thiết bị đánh giá: {device}")

    base_model = TransMambaLong()
    lora_config = LoraConfig(
        r=8, lora_alpha=16, 
        target_modules=["query", "value"], 
        lora_dropout=0.1, bias="none"
    )
    model = get_peft_model(base_model, lora_config).to(device)
    
    try:
        model.load_state_dict(torch.load("transmamba_long_weights.pth", map_location=device))
    except FileNotFoundError:
        print("[-] Cảnh báo: Đang chạy đánh giá mà không có file trọng số .pth đã train.")
        
    model.eval()

    print("[*] Đang tải tập Test...")
    _, test_loader = get_dataloaders(batch_size=4, max_length=1024)

    all_preds = []
    all_labels = []

    print("[*] BẮT ĐẦU CHẤM ĐIỂM...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Đang đánh giá"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(ids, mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print("\n" + "="*60)
    print("📊 BẢNG KẾT QUẢ ĐÁNH GIÁ (EVALUATION METRICS)")
    print("="*60)
    print(f"🎯 Accuracy (Độ chính xác tổng thể): {acc:.4f} ({acc*100:.2f}%)")
    print("-" * 60)
    print("📌 Chi tiết các chỉ số:")
    print(classification_report(all_labels, all_preds, target_names=["Tiêu cực (Neg)", "Tích cực (Pos)"], digits=4))
    print("="*60)

if __name__ == "__main__":
    main()