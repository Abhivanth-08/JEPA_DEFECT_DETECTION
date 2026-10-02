import sys
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.decomposition import PCA
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
PCA_FILE = cfg.CHECKPOINTS_DIR / 'pca_reducer.joblib'

class EnergyModel(nn.Module):

    def __init__(self, input_dim: int=cfg.SVDD_DIM, proj_dim: int=cfg.SVDD_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.05), nn.Linear(128, proj_dim))
        self.register_buffer('center', torch.zeros(proj_dim))
        self._center_set = False
        self.pca_dim = input_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return ((z - self.center) ** 2).sum(dim=-1)

    @staticmethod
    def fit_pca(embeddings: np.ndarray, max_components: int=64) -> PCA:
        N, D = embeddings.shape
        n_comp = min(N - 2, max_components, D)
        print(f'[PCA] {D}-D → {n_comp}-D (from {N} samples)')
        pca = PCA(n_components=n_comp)
        pca.fit(embeddings)
        var_explained = pca.explained_variance_ratio_.sum() * 100
        print(f'[PCA] Explained variance: {var_explained:.1f}%')
        return pca

    @staticmethod
    def save_pca(pca: PCA, path: Path=PCA_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pca, str(path))
        print(f'[PCA] Saved to {path.name}')

    @staticmethod
    def load_pca(path: Path=PCA_FILE) -> Optional[PCA]:
        if not path.exists():
            return None
        pca = joblib.load(str(path))
        print(f'[PCA] Loaded ({pca.n_components_} components)')
        return pca

    @torch.no_grad()
    def fit_center(self, embeddings_pca: np.ndarray, device: str=cfg.DEVICE, batch_size: int=256) -> None:
        self.eval()
        self.to(device)
        all_z = []
        t = torch.from_numpy(embeddings_pca.astype(np.float32)).to(device)
        for i in range(0, len(t), batch_size):
            z = self.encode(t[i:i + batch_size])
            all_z.append(z)
        c = torch.cat(all_z, 0).mean(0)
        c[(c.abs() < 0.01) & (c > 0)] = 0.01
        c[(c.abs() < 0.01) & (c < 0)] = -0.01
        self.center.copy_(c)
        self._center_set = True
        print(f'[SVDD] Center set | norm={c.norm().item():.4f} | dim={len(c)}')

    @torch.no_grad()
    def score_numpy(self, cls_emb_pca: np.ndarray, device: str=cfg.DEVICE) -> float:
        self.eval()
        self.to(device)
        t = torch.from_numpy(cls_emb_pca.astype(np.float32)).unsqueeze(0).to(device) if cls_emb_pca.ndim == 1 else torch.from_numpy(cls_emb_pca.astype(np.float32)).to(device)
        e = self.forward(t)
        return e.squeeze().cpu().item() if cls_emb_pca.ndim == 1 else e.cpu().numpy()

    def save(self, path: Path):
        torch.save({'state_dict': self.state_dict(), 'center': self.center.cpu().numpy(), 'pca_dim': self.pca_dim}, str(path))

    def load(self, path: Path, device: str=cfg.DEVICE):
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        self.load_state_dict(ckpt['state_dict'])
        self.center.copy_(torch.from_numpy(ckpt['center']).to(device))
        self._center_set = True
        self.to(device)