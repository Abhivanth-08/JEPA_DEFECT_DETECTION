import sys
from pathlib import Path
from collections import deque
from typing import Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from models.temporal_transformer import TemporalTransformer
from models.spatial_jepa import SpatialJEPAHead
from anomaly.energy_model import EnergyModel

class AnomalyScorer:

    def __init__(self, temporal_short: TemporalTransformer, temporal_long: TemporalTransformer, spatial_head: SpatialJEPAHead, energy_model: EnergyModel, device: str=cfg.DEVICE, alpha: float=cfg.SCORE_ALPHA, alpha_long: float=cfg.SCORE_ALPHA_LONG, beta: float=cfg.SCORE_BETA, gamma: float=cfg.SCORE_GAMMA, delta: float=cfg.SCORE_DELTA, t_scale: float=1.0, t_long_scale: float=1.0, s_scale: float=1.0, e_scale: float=1.0, p_scale: float=1.0, mc_passes: int=cfg.MC_DROPOUT_PASSES, pca_model=None, normal_cls_bank: Optional[np.ndarray]=None, normal_patch_bank: Optional[np.ndarray]=None):
        self.device = device
        self.temporal_s = temporal_short.to(device)
        self.temporal_l = temporal_long.to(device)
        self.spatial = spatial_head.to(device)
        self.energy = energy_model.to(device)
        self.alpha = alpha
        self.alpha_long = alpha_long
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.t_scale = t_scale
        self.t_long_scale = t_long_scale
        self.s_scale = s_scale
        self.e_scale = e_scale
        self.p_scale = p_scale
        self.mc_passes = mc_passes
        self.pca_model = pca_model
        self.normal_cls_bank = normal_cls_bank
        self.normal_patch_bank = normal_patch_bank
        self.short_buf: deque = deque(maxlen=cfg.WINDOW_SIZE)
        self.long_buf: deque = deque(maxlen=cfg.LONG_WINDOW_SIZE)
        self.temporal_s.eval()
        self.temporal_l.eval()
        self.spatial.eval()
        self.energy.eval()

    def reset(self):
        self.short_buf.clear()
        self.long_buf.clear()

    def is_ready(self) -> bool:
        return len(self.short_buf) >= cfg.WINDOW_SIZE

    def _patch_nn_score(self, cls_emb: np.ndarray, patch_emb: np.ndarray) -> float:
        if self.normal_cls_bank is None or self.normal_patch_bank is None:
            return 0.0
        a = cls_emb.flatten()
        norms_a = np.linalg.norm(a) + 1e-08
        norms_n = np.linalg.norm(self.normal_cls_bank, axis=1) + 1e-08
        sims = self.normal_cls_bank @ a / (norms_n * norms_a)
        sorted_idx = np.argsort(-sims)
        best_idx = None
        for idx in sorted_idx:
            if sims[idx] < 0.999:
                best_idx = idx
                break
        if best_idx is None:
            best_idx = sorted_idx[-1]
        normal_patches = self.normal_patch_bank[best_idx]
        min_p = min(len(patch_emb), len(normal_patches))
        test_p = patch_emb[:min_p]
        norm_p = normal_patches[:min_p]
        dot = np.sum(test_p * norm_p, axis=-1)
        norm_t = np.linalg.norm(test_p, axis=-1) + 1e-08
        norm_n = np.linalg.norm(norm_p, axis=-1) + 1e-08
        cos_sim = dot / (norm_t * norm_n)
        cos_dist = 1.0 - cos_sim
        return float(cos_dist.mean())

    def push_and_score(self, cls_emb: np.ndarray, patch_emb: np.ndarray) -> Tuple[float, Dict[str, float]]:
        self.short_buf.append(cls_emb.copy())
        self.long_buf.append(cls_emb.copy())
        if not self.is_ready():
            return (0.0, {'temporal': 0.0, 'temporal_long': 0.0, 'spatial': 0.0, 'energy': 0.0, 'patch_nn': 0.0, 'uncertainty': 0.0})
        cls_t = torch.from_numpy(cls_emb.astype(np.float32)).unsqueeze(0).to(self.device)
        patch_t = torch.from_numpy(patch_emb.astype(np.float32)).unsqueeze(0).to(self.device)
        ctx_short = torch.from_numpy(np.stack(list(self.short_buf), axis=0).astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean_pred_s, variance_s = self.temporal_s.mc_forward(ctx_short, self.mc_passes)
        t_err_short = F.mse_loss(mean_pred_s, cls_t).item() + (1.0 - F.cosine_similarity(mean_pred_s, cls_t, dim=-1).mean()).item()
        unc = variance_s.item()
        t_score_short = t_err_short / self.t_scale
        t_score_long = 0.0
        if len(self.long_buf) >= cfg.LONG_WINDOW_SIZE:
            long_list = list(self.long_buf)
            downsampled = long_list[::cfg.LONG_DOWNSAMPLE]
            target_len = max(cfg.LONG_WINDOW_SIZE // cfg.LONG_DOWNSAMPLE, 1)
            if len(downsampled) < target_len:
                pad = [downsampled[0]] * (target_len - len(downsampled))
                downsampled = pad + downsampled
            else:
                downsampled = downsampled[-target_len:]
            ctx_long = torch.from_numpy(np.stack(downsampled, axis=0).astype(np.float32)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred_l = self.temporal_l(ctx_long)
            t_err_long = F.mse_loss(pred_l, cls_t).item() + (1.0 - F.cosine_similarity(pred_l, cls_t, dim=-1).mean()).item()
            t_score_long = t_err_long / self.t_long_scale
        P = patch_t.size(1)
        n_masked = int(P * cfg.MASK_RATIO)
        start = (P - n_masked) // 2
        mask = torch.zeros(1, P, dtype=torch.bool, device=self.device)
        mask[0, start:start + n_masked] = True
        masked_patches = patch_t.clone()
        masked_patches[mask.unsqueeze(-1).expand_as(masked_patches)] = 0.0
        with torch.no_grad():
            pred_masked = self.spatial(masked_patches, mask)
            true_masked = patch_t[mask.unsqueeze(-1).expand_as(patch_t)].view(1, n_masked, -1)
        s_err = F.mse_loss(pred_masked, true_masked[:, :pred_masked.size(1)]).item()
        s_score = s_err / self.s_scale
        e_score = 0.0
        if self.pca_model is not None:
            cls_pca = self.pca_model.transform(cls_emb.reshape(1, -1))
            cls_pca_t = torch.from_numpy(cls_pca.astype(np.float32)).to(self.device)
            with torch.no_grad():
                e_val = self.energy(cls_pca_t).item()
            e_score = e_val / self.e_scale
        else:
            with torch.no_grad():
                e_val = self.energy(cls_t).item()
            e_score = e_val / self.e_scale
        p_score_raw = self._patch_nn_score(cls_emb, patch_emb)
        p_score = p_score_raw / self.p_scale
        score = self.alpha * t_score_short + self.alpha_long * t_score_long + self.beta * s_score + self.gamma * e_score + self.delta * p_score
        return (score, {'temporal': t_score_short, 'temporal_long': t_score_long, 'spatial': s_score, 'energy': e_score, 'patch_nn': p_score, 'uncertainty': unc})