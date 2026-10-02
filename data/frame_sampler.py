import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple
import sys, os
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

def extract_frames(video_path: str | Path, output_dir: str | Path, target_fps: int=cfg.TARGET_FPS, frame_size: int=cfg.FRAME_SIZE, progress_callback=None, human_masker=None) -> list[Path]:
    video_path = Path(video_path)
    output_dir = Path(output_dir) / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / target_fps))
    saved_paths = []
    src_idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if src_idx % step == 0:
            frame_resized = cv2.resize(frame, (frame_size, frame_size))
            
            # 1. Optional Grayscale
            if cfg.USE_GRAYSCALE:
                gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                frame_resized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # 2. Global Contrast Normalization (GCN)
            if cfg.GCN_ENABLE:
                f_float = frame_resized.astype(np.float32)
                mean, std = cv2.meanStdDev(f_float)
                mean, std = mean.flatten(), std.flatten()
                f_float = (f_float - mean) / (std + 1e-6)
                # Rescale back to 0-255 for CLAHE
                frame_resized = cv2.normalize(f_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # 3. CLAHE
            lab = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            frame_norm = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            if human_masker is not None:
                frame_norm = human_masker.mask(frame_norm)
            out_path = output_dir / f'frame_{saved_idx:06d}.jpg'
            cv2.imwrite(str(out_path), frame_norm)
            saved_paths.append(out_path)
            saved_idx += 1
            if progress_callback and total_src > 0:
                progress_callback(src_idx, total_src)
        src_idx += 1
    cap.release()
    return saved_paths

def load_frames_as_numpy(frame_paths: list[Path]) -> np.ndarray:
    frames = []
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0) if frames else np.empty((0, cfg.FRAME_SIZE, cfg.FRAME_SIZE, 3), dtype=np.uint8)

def get_saved_frame_paths(video_stem: str, frames_dir: Path=cfg.FRAMES_DIR) -> list[Path]:
    d = frames_dir / video_stem
    if not d.exists():
        return []
    return sorted(d.glob('frame_*.jpg'))