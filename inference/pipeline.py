import sys
from pathlib import Path
from typing import Callable, Optional, List, Dict
import cv2
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from models.encoder import ViTEncoder
from models.temporal_transformer import TemporalTransformer
from models.spatial_jepa import SpatialJEPAHead
from anomaly.energy_model import EnergyModel
from anomaly.scorer import AnomalyScorer
from training.calibration import load_calibration

def build_scorer(device: str=cfg.DEVICE, video_stem: str=''):
    long_seq_len = max(cfg.LONG_WINDOW_SIZE // cfg.LONG_DOWNSAMPLE, 4)
    temporal_short = TemporalTransformer(window_size=cfg.WINDOW_SIZE).to(device)
    temporal_short.load(cfg.CHECKPOINTS_DIR / 'temporal.pt', device)
    temporal_long = TemporalTransformer(window_size=long_seq_len).to(device)
    if (cfg.CHECKPOINTS_DIR / 'temporal_long.pt').exists():
        temporal_long.load(cfg.CHECKPOINTS_DIR / 'temporal_long.pt', device)
    spatial_head = SpatialJEPAHead().to(device)
    spatial_head.load(cfg.CHECKPOINTS_DIR / 'spatial.pt', device)
    pca_model = EnergyModel.load_pca()
    if pca_model is not None:
        pca_dim = pca_model.n_components_
        energy_model = EnergyModel(input_dim=pca_dim, proj_dim=cfg.SVDD_DIM).to(device)
    else:
        energy_model = EnergyModel(input_dim=cfg.EMBED_DIM, proj_dim=cfg.SVDD_DIM).to(device)
    if cfg.ENERGY_FILE.exists():
        energy_model.load(cfg.ENERGY_FILE, device)
    encoder = ViTEncoder(device=device)
    normal_cls_bank = None
    normal_patch_bank = None
    stems_to_try = [video_stem] if video_stem else []
    if not stems_to_try:
        for f in cfg.EMBEDDINGS_DIR.glob('*_cls.npy'):
            stems_to_try.append(f.stem.replace('_cls', ''))
    for stem in stems_to_try:
        cls_p = cfg.EMBEDDINGS_DIR / f'{stem}_cls.npy'
        patch_p = cfg.EMBEDDINGS_DIR / f'{stem}_patches.npy'
        if cls_p.exists() and patch_p.exists():
            normal_cls_bank = np.load(str(cls_p))
            normal_patch_bank = np.load(str(patch_p))
            print(f"[Scorer] Loaded normal bank: {normal_cls_bank.shape[0]} frames from '{stem}'")
            break
    calib = load_calibration() or {}
    threshold = calib.get('threshold', 1.0)
    t_scale = calib.get('t_scale', 1.0)
    t_long_sc = calib.get('t_long_scale', 1.0)
    s_scale = calib.get('s_scale', 1.0)
    e_scale = calib.get('e_scale', 1.0)
    p_scale = calib.get('p_scale', 1.0)
    scorer = AnomalyScorer(temporal_short=temporal_short, temporal_long=temporal_long, spatial_head=spatial_head, energy_model=energy_model, device=device, t_scale=t_scale, t_long_scale=t_long_sc, s_scale=s_scale, e_scale=e_scale, p_scale=p_scale, pca_model=pca_model, normal_cls_bank=normal_cls_bank, normal_patch_bank=normal_patch_bank)
    return (scorer, encoder, threshold)

def run_video_inference(video_path: Path, scorer: AnomalyScorer, encoder: ViTEncoder, threshold: float, frame_callback: Optional[Callable]=None, max_frames: Optional[int]=None, human_masker=None) -> List[Dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {video_path}')
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / cfg.TARGET_FPS))
    scorer.reset()
    results = []
    raw_idx = 0
    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break
        raw_idx += 1
        if (raw_idx - 1) % step != 0:
            continue
        frame_resized = cv2.resize(frame_bgr, (cfg.FRAME_SIZE, cfg.FRAME_SIZE))
        
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
            frame_resized = cv2.normalize(f_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 3. CLAHE
        lab = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        frame_norm = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        _, enc_jpg = cv2.imencode('.jpg', frame_norm, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        frame_norm = cv2.imdecode(enc_jpg, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame_norm, cv2.COLOR_BGR2RGB)
        display_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        encode_frame = frame_rgb
        if human_masker is not None:
            masked_bgr = human_masker.mask(frame_norm)
            encode_frame = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)
        cls_emb, patch_emb = encoder.encode_frame_np(encode_frame)
        score, components = scorer.push_and_score(cls_emb, patch_emb)
        is_anomaly = score > threshold
        result = {'frame_idx': frame_idx, 'score': score, 'is_anomaly': is_anomaly, 'frame_rgb': display_frame, **components}
        results.append(result)
        if frame_callback:
            frame_callback(display_frame, result, frame_idx)
        frame_idx += 1
    cap.release()
    return results