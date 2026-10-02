import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

class EmbeddingWindowDataset(Dataset):

    def __init__(self, embeddings: np.ndarray, window_size: int=cfg.WINDOW_SIZE):
        super().__init__()
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.window_size = window_size
        self.indices = list(range(window_size, len(embeddings)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t = self.indices[idx]
        context = self.embeddings[t - self.window_size:t]
        target = self.embeddings[t]
        return (context, target)

class PatchEmbeddingDataset(Dataset):

    def __init__(self, patch_embeddings: np.ndarray, mask_ratio: float=cfg.MASK_RATIO):
        super().__init__()
        self.patches = torch.tensor(patch_embeddings, dtype=torch.float32)
        self.mask_ratio = mask_ratio
        self.N, self.P, self.D = self.patches.shape

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        patches = self.patches[idx]
        num_masked = int(self.P * self.mask_ratio)
        start = np.random.randint(0, self.P - num_masked + 1)
        mask = torch.zeros(self.P, dtype=torch.bool)
        mask[start:start + num_masked] = True
        visible_patches = patches.clone()
        visible_patches[mask] = 0.0
        target_patches = patches[mask]
        return (visible_patches, target_patches, mask)