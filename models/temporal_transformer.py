import sys, math
from pathlib import Path
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int=512, dropout: float=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])

class TemporalTransformer(nn.Module):

    def __init__(self, embed_dim: int=cfg.EMBED_DIM, window_size: int=cfg.WINDOW_SIZE, num_layers: int=cfg.T_NUM_LAYERS, num_heads: int=cfg.T_NUM_HEADS, ff_dim: int=cfg.T_FF_DIM, dropout: float=cfg.MC_DROPOUT_RATE):
        super().__init__()
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=512, dropout=0.0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.predictor = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim))
        causal_mask = torch.triu(torch.full((window_size, window_size), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', causal_mask)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(x)
        h = self.transformer(x, mask=self.causal_mask)
        return self.predictor(h[:, -1])

    def mc_forward(self, x: torch.Tensor, n_samples: int=cfg.MC_DROPOUT_PASSES) -> Tuple[torch.Tensor, torch.Tensor]:
        self.train()
        preds = torch.stack([self.forward(x) for _ in range(n_samples)], dim=0)
        self.eval()
        mean_pred = preds.mean(dim=0)
        variance = preds.var(dim=0).mean(dim=-1)
        return (mean_pred, variance)

    def save(self, path: Path):
        torch.save({'state_dict': self.state_dict(), 'window_size': self.window_size}, str(path))

    def load(self, path: Path, device: str='cpu'):
        ckpt = torch.load(str(path), map_location=device)
        self.load_state_dict(ckpt['state_dict'])
        self.to(device)