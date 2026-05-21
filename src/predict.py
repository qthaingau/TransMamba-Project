import torch
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from model import TransMambaLong

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    base_model = TransMambaLong()
    lora_config = LoraConfig(
        r=8, lora_alpha=16, 
        target_modules=["query", "value"], 
        lora_dropout=0.1, bias="none"
    )
    model = get_peft_model(base_model, lora_config).to(device)

    weights_path = "transmamba_long_weights.pth"
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("[+] Nạp bộ nhớ thành công!")
    except FileNotFoundError:
        print("[-] Cảnh báo: Không tìm thấy file .pth. Mô hình sẽ dự đoán ngẫu nhiên.")
        
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

    def predict_review(text):
        inputs = tokenizer(
            text, padding="max_length", truncation=True, max_length=1024, return_tensors="pt"
        )
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            logits = model(ids, mask)
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze()
            pred_class = torch.argmax(probs).item()
        
        labels = ["Tiêu cực (Negative)", "Tích cực (Positive)"]
        print(f"\n" + "="*50)
        print(f"📄 NỘI DUNG:\n'{text}'")
        print(f"🤖 DỰ ĐOÁN: >> {labels[pred_class]} <<")
        print(f"📊 Tự tin: Tiêu cực ({probs[0]:.2%}) | Tích cực ({probs[1]:.2%})")
        print("="*50)

    print("\n[+] HỆ THỐNG INFERENCE SẴN SÀNG...")
    review_1 = "This movie is an absolute masterpiece. The acting is brilliant and the plot is deep."
    predict_review(review_1)
    
    review_2 = "What a complete waste of time. The storyline was boring."
    predict_review(review_2)

if __name__ == "__main__":
    main()