import sys
from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

class SpatialJEPAHead(nn.Module):

    def __init__(self, embed_dim: int=cfg.EMBED_DIM, num_heads: int=cfg.SPATIAL_NUM_HEADS, num_enc_layers: int=cfg.SPATIAL_NUM_ENC_LAYERS, num_dec_layers: int=cfg.SPATIAL_NUM_DEC_LAYERS, dropout: float=cfg.T_DROPOUT):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2, dropout=dropout, batch_first=True, norm_first=True)
        self.context_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        dec_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2, dropout=dropout, batch_first=True, norm_first=True)
        self.predictor = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, patches: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, P, D = patches.shape
        mask_bool = mask.bool()
        vis_mask = ~mask_bool
        context = self.context_encoder(patches, src_key_padding_mask=mask_bool)
        mask_tokens = self.mask_token.expand(B, P, D)
        query = torch.where(mask_bool.unsqueeze(-1), mask_tokens, patches)
        pred_all = self.predictor(tgt=query, memory=context, tgt_key_padding_mask=~mask_bool)
        pred_all = self.out_proj(self.out_norm(pred_all))
        n_masked = mask_bool[0].sum().item()
        pred_masked = pred_all[mask_bool].view(B, int(n_masked), D)
        return pred_masked

    def reconstruct_error(self, patches: torch.Tensor, mask: torch.Tensor, true_patches: torch.Tensor) -> torch.Tensor:
        pred = self.forward(patches, mask)
        n = min(pred.size(1), true_patches.size(1))
        return torch.nn.functional.mse_loss(pred[:, :n], true_patches[:, :n])

    def save(self, path: Path):
        torch.save(self.state_dict(), str(path))

    def load(self, path: Path, device: str='cpu'):
        self.load_state_dict(torch.load(str(path), map_location=device))
        self.to(device)