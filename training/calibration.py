import json, sys
import numpy as np
from pathlib import Path
from typing import Optional
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

def calibrate_threshold(normal_scores: np.ndarray, component_scales: Optional[dict]=None, percentile: int=cfg.CALIBRATION_PERCENTILE) -> float:
    raw_threshold = float(np.percentile(normal_scores, percentile))
    margin = max(float(normal_scores.std()) * 0.5, raw_threshold * 0.1)
    threshold = raw_threshold + margin
    data: dict = {'threshold': threshold, 'mean': float(normal_scores.mean()), 'std': float(normal_scores.std()), 'percentile': percentile}
    if component_scales:
        data.update(component_scales)
    cfg.CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    cfg.CALIBRATION_FILE.write_text(json.dumps(data, indent=2))
    return threshold

def calibrate_component_scales(t_errors: np.ndarray, t_long_errors: np.ndarray, s_errors: np.ndarray, e_scores: np.ndarray, p_errors: np.ndarray=None, pct: int=95) -> dict:

    def safe_pct(arr):
        v = float(np.percentile(arr[arr > 0], pct)) if (arr > 0).any() else 1.0
        return max(v, 0.001)
    result = {'t_scale': safe_pct(t_errors), 't_long_scale': safe_pct(t_long_errors) if len(t_long_errors) > 0 else 1.0, 's_scale': safe_pct(s_errors), 'e_scale': safe_pct(e_scores)}
    if p_errors is not None:
        result['p_scale'] = safe_pct(p_errors)
    else:
        result['p_scale'] = 1.0
    return result

def load_calibration() -> Optional[dict]:
    if not cfg.CALIBRATION_FILE.exists():
        return None
    return json.loads(cfg.CALIBRATION_FILE.read_text())

def load_threshold() -> float:
    c = load_calibration()
    return c['threshold'] if c else 1.0