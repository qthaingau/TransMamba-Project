import torch
import torch.nn as nn
from transformers import AutoModel
from mamba_ssm import Mamba

class TransMambaLong(nn.Module):
    """
    Kiến trúc TransMamba-Long (Phiên bản 4 Lớp).
    Kết hợp Longformer Encoder và Mamba Decoder (O(n) complexity).
    """
    def __init__(self, encoder_name="allenai/longformer-base-4096", d_model=768, num_classes=2):
        super().__init__()
        
        # 1. Encoder: Longformer (Trích xuất đặc trưng cục bộ)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        
        # 2. Decoder: 4 khối Mamba (Tối ưu tốc độ, Linear-time)
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) 
            for _ in range(4) # <-- Khóa cứng 4 lớp
        ])
        self.mamba_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])

        # 3. Fusion: Trộn đặc trưng qua Multi-Head Cross-Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        self.final_norm = nn.LayerNorm(d_model)

        # 4. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        # Lấy hidden states từ Encoder
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # Đẩy qua Mamba Decoder
        h = enc_out
        for mamba, norm in zip(self.mamba_layers, self.mamba_norms):
            h = mamba(norm(h)) + h
            
        # Cross-Attention Fusion
        fused, _ = self.cross_attn(query=h, key=enc_out, value=enc_out)
        out = self.final_norm(h + fused)
        
        # Pooling và Phân loại
        pooled = out.mean(dim=1) 
        return self.classifier(pooled)