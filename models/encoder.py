import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from typing import Tuple, List
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

class ViTEncoder(nn.Module):

    def __init__(self, model_name: str=cfg.ENCODER_MODEL, device: str=cfg.DEVICE, finetune: bool=cfg.ENCODER_FINETUNE, freeze_blocks: int=cfg.ENCODER_FREEZE_BLOCKS):
        super().__init__()
        self.device = device
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device)
        for param in self.model.parameters():
            param.requires_grad = False
        if finetune:
            self.partial_unfreeze(freeze_blocks)
        ft_path = cfg.CHECKPOINTS_DIR / 'encoder_finetune.pt'
        if ft_path.exists():
            self._load_finetuned(ft_path, device)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)])
        self.eval()

    def partial_unfreeze(self, n_freeze: int=8) -> None:
        total = len(self.model.blocks)
        for i, block in enumerate(self.model.blocks):
            for param in block.parameters():
                param.requires_grad = i >= n_freeze
        for param in self.model.norm.parameters():
            param.requires_grad = True
        n_unfrozen = total - n_freeze
        trainable = sum((p.numel() for p in self.model.parameters() if p.requires_grad))
        total_p = sum((p.numel() for p in self.model.parameters()))
        print(f'[Encoder] {n_freeze}/{total} blocks frozen | {n_unfrozen} blocks trainable ({trainable:,} / {total_p:,} params, {100 * trainable / total_p:.1f}% unfrozen)')

    def trainable_parameters(self):
        return (p for p in self.model.parameters() if p.requires_grad)

    def _load_finetuned(self, path: Path, device: str):
        try:
            saved = torch.load(str(path), map_location=device)
            loaded = 0
            model_dict = dict(self.model.named_parameters())
            for name, param_data in saved.items():
                clean = name.replace('model.', '') if name.startswith('model.') else name
                if clean in model_dict:
                    model_dict[clean].data.copy_(param_data.data)
                    loaded += 1
            print(f'[Encoder] Loaded {loaded} fine-tuned parameter tensors from {path.name}')
        except Exception as e:
            print(f'[Encoder] Warning: could not load fine-tuned weights: {e}')

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.model.forward_features(pixel_values)
        cls_tokens = features[:, 0]
        patch_tokens = features[:, 1:]
        return (cls_tokens, patch_tokens)

    @torch.no_grad()
    def encode_frame_np(self, frame_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        cls, patches = self.forward(t)
        return (cls[0].cpu().numpy(), patches[0].cpu().numpy())

    @torch.no_grad()
    def encode_batch_numpy(self, frames: np.ndarray, batch_size: int=16) -> Tuple[np.ndarray, np.ndarray]:
        all_cls, all_patches = ([], [])
        for i in range(0, len(frames), batch_size):
            batch_np = frames[i:i + batch_size]
            tensors = torch.stack([self.transform(f) for f in batch_np]).to(self.device)
            cls, pat = self.forward(tensors)
            all_cls.append(cls.cpu().numpy())
            all_patches.append(pat.cpu().numpy())
        return (np.concatenate(all_cls, 0), np.concatenate(all_patches, 0))