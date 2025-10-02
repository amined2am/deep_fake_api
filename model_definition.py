# model_definition.py
import math
import torch
from torch import nn
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                         # [1, max_len, d_model]
        self.register_buffer("pe", pe)               # => attention.positional_encoding.pe

    def forward(self, x):                             # x: [B, T, d_model]
        T = x.size(1)
        return x + self.pe[:, :T, :]

class TemporalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, max_len: int = 200):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):                             # [B, T, d_model]
        x = self.positional_encoding(x)
        attn_out, _ = self.multihead_attn(x, x, x)
        x = self.norm(x + attn_out)
        return x

class Model(nn.Module):
    def __init__(self, num_classes: int = 2, attn_max_len: int = 200, attn_heads: int = 8):
        super().__init__()
        backbone = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.lstm = nn.LSTM(input_size=2048, hidden_size=512, num_layers=2, bidirectional=True, batch_first=True)

        d_model = 1024  # 512 * 2
        self.attention = TemporalAttention(d_model=d_model, num_heads=attn_heads, max_len=attn_max_len)

        self.fc1 = nn.Linear(d_model, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):                              # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        fmap = self.feature_extractor(x)               # [B*T, 2048, H', W']
        x = self.avgpool(fmap)                         # [B*T, 2048, 1, 1]
        x = x.view(b, t, -1)                           # [B, T, 2048]
        x_lstm, _ = self.lstm(x)                       # [B, T, 1024]
        x_attn = self.attention(x_lstm)                # [B, T, 1024]
        x_seq = torch.mean(x_attn, dim=1)              # [B, 1024]
        x = self.dropout(torch.relu(self.fc1(x_seq)))  # [B, 512]
        logits = self.fc2(x)                           # [B, 2]
        return fmap, logits
