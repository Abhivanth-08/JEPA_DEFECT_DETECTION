import torch
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

def temporal_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    cos = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    return mse + cos

def spatial_loss(pred_masked: torch.Tensor, true_masked: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_masked, true_masked)

def combined_loss(t_loss: torch.Tensor, s_loss: torch.Tensor, alpha: float=cfg.LOSS_ALPHA, beta: float=cfg.LOSS_BETA) -> torch.Tensor:
    return alpha * t_loss + beta * s_loss

def svdd_loss(projections: torch.Tensor, center: torch.Tensor, nu: float=cfg.SVDD_NU) -> torch.Tensor:
    dist = ((projections - center) ** 2).sum(dim=-1)
    r_sq = dist.mean().detach()
    penalty = F.relu(dist - r_sq).mean()
    return dist.mean() + 1.0 / max(nu, 1e-06) * penalty