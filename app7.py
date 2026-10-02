

import sys, json, time, tempfile, shutil, math
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg




st.set_page_config(
    page_title="JEPA Anomaly Detector v7",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)




st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp { background: #ffffff; }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f0f4fa 0%, #e8eef8 100%);
    border-right: 1px solid #d0dcea;
  }

  .metric-card {
    background: #f5f8ff;
    border: 1px solid #c8d8f0;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(80,120,200,0.08);
  }
  .metric-card .value { font-size: 2rem; font-weight: 700; }
  .metric-card .label { font-size: 0.82rem; color: #4a6a9a; margin-top: 4px; }

  .alert-normal {
    background: rgba(0,168,84,0.08);
    border-left: 4px solid #00a854;
    border-radius: 8px; padding: 12px 18px;
    color: #007a3d; font-weight: 600; font-size: 1.05rem;
  }
  .alert-anomaly {
    background: rgba(220,50,50,0.1);
    border-left: 4px solid #e03030;
    border-radius: 8px; padding: 12px 18px;
    color: #c02020; font-weight: 600; font-size: 1.05rem;
    animation: pulse 1s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.65} }

  .phase-pill {
    display: inline-block;
    background: linear-gradient(90deg,#1a6ab5,#2a85d5);
    border-radius: 20px; padding: 4px 14px;
    font-size: 0.72rem; font-weight: 600;
    color: #fff; letter-spacing: .05em; text-transform: uppercase;
  }

  h1 { color: #1a2a4a !important; font-weight: 700 !important; }
  h2 { color: #2a4a7a !important; font-weight: 600 !important; }
  h3 { color: #3a5a8a !important; }

  .stProgress > div > div { background: linear-gradient(90deg,#1a6ab5,#00a8e8) !important; }

  .stTabs [data-baseweb="tab-list"] { gap: 10px; background: transparent; }
  .stTabs [data-baseweb="tab"] {
    background: #eef3fb; border: 1px solid #c8d8f0;
    border-radius: 10px; color: #3a5a8a; font-weight: 600; padding: 10px 22px;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#1a6ab5,#0d4d8a);
    border-color: #1a6ab5; color: #fff !important;
  }

  .stButton > button {
    background: linear-gradient(135deg,#1a6ab5,#0d4d8a);
    color: white; border: none; border-radius: 10px;
    font-weight: 600; padding: 11px 26px; font-size: .95rem; transition: all .2s;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg,#2280d5,#1260aa);
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(26,106,181,.25);
  }

  label { color: #3a5a8a !important; }
  .upload-card {
    background: #f5f8ff; border: 2px dashed #c8d8f0;
    border-radius: 16px; padding: 28px; text-align: center;
    margin-bottom: 12px;
  }
</style>
""", unsafe_allow_html=True)





def _ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

_ss("normal_video_path", None)
_ss("normal_video_stem", None)
_ss("trained", False)
_ss("calibrated", False)
_ss("threshold", 1.0)
_ss("trainer_obj", None)
_ss("webcam_running", False)





UPLOADS_DIR = Path(__file__).parent / "user_uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_video(uploaded_file) -> Path:
    dest = UPLOADS_DIR / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def calculate_confidence(uncertainty: float) -> float:
    """Calculates confidence percentage based on MC Dropout Epistemic Uncertainty."""
    # Typical MC dropout variance is quite low.
    # We use an exponential decay so high variance = low confidence.
    conf = math.exp(-uncertainty * 50.0) * 100.0
    return 100.0 - max(0.0, min(100.0, conf))

def score_to_color(score: float, threshold: float) -> str:
    r = score / max(threshold, 1e-6)
    if r < 1.0:  return "#00a854"
    return "#e03030"


def make_loss_curve(loss_history: list) -> go.Figure:
    fig = go.Figure(go.Scatter(
        y=loss_history, mode="lines", line=dict(color="#1a6ab5", width=2)
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f5f8ff",
        xaxis=dict(title="Epoch", color="#3a5a8a", gridcolor="#dde8f5"),
        yaxis=dict(title="Loss",  color="#3a5a8a", gridcolor="#dde8f5"),
        margin=dict(l=10, r=10, t=10, b=10), height=220,
        font=dict(color="#3a5a8a"),
    )
    return fig


def make_score_timeline(results: list, threshold: float) -> go.Figure:
    frames = [r["frame_idx"] for r in results]
    scores = [r["score"]     for r in results]
    colors = [score_to_color(s, threshold) for s in scores]
    texts  = [f"Score: {s:.3f}<br>Conf: {r.get('confidence', 0.0):.1f}%" for s, r in zip(scores, results)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frames, y=scores, mode="lines+markers",
        line=dict(color="#1a6ab5", width=2),
        marker=dict(color=colors, size=6),
        name="Anomaly Score",
        text=texts,
        hoverinfo="x+text",
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#e03030",
                  line_width=2, annotation_text="Anomaly Threshold (Red)",
                  annotation_font_color="#e03030")
    for r in [r for r in results if r["is_anomaly"]]:
        fig.add_vrect(x0=r["frame_idx"]-.5, x1=r["frame_idx"]+.5,
                      fillcolor="rgba(224,48,48,0.08)", line_width=0)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f5f8ff",
        xaxis=dict(title="Frame", color="#3a5a8a", gridcolor="#dde8f5"),
        yaxis=dict(title="Score", color="#3a5a8a", gridcolor="#dde8f5"),
        margin=dict(l=10, r=10, t=10, b=10), height=260,
        showlegend=False, font=dict(color="#3a5a8a"),
    )
    return fig


def make_gauge(score: float, threshold: float) -> go.Figure:
    color = score_to_color(score, threshold)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 4),
        number={"font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [0, threshold * 2], "tickcolor": "#3a5a8a",
                     "tickfont": {"color": "#3a5a8a"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0,         threshold],   "color": "rgba(0,168,84,0.10)"},
                {"range": [threshold, threshold*2], "color": "rgba(224,48,48,0.12)"},
            ],
            "threshold": {"line": {"color": "#555", "width": 3},
                          "thickness": 0.8, "value": threshold},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=30, b=20),
        height=190, font={"color": "#3a5a8a"},
    )
    return fig


def make_component_bar(components: dict) -> go.Figure:
    
    keys   = ["T-Short", "T-Long", "Spatial", "Energy", "Uncertainty"]
    vals   = [
        components.get("temporal",      0),
        components.get("temporal_long", 0),
        components.get("spatial",       0),
        components.get("energy",        0),
        components.get("uncertainty",   0),
    ]
    colors = ["#1a6ab5", "#00a8e8", "#7c4fff", "#ff6b35", "#00a854"]
    fig = go.Figure(go.Bar(x=keys, y=vals, marker_color=colors,
                           marker_line_color="rgba(0,0,0,0)"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f5f8ff",
        xaxis=dict(color="#3a5a8a"), yaxis=dict(color="#3a5a8a", gridcolor="#dde8f5"),
        margin=dict(l=10, r=10, t=10, b=10), height=190,
        font=dict(color="#3a5a8a"),
    )
    return fig


def annotate_frame(frame_rgb: np.ndarray, label: str,
                   score=None, conf=None, is_anomaly: bool = False) -> np.ndarray:
    bgr = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
    bc  = (0, 60, 255) if is_anomaly else (0, 180, 80)
    cv2.rectangle(bgr, (0,0), (bgr.shape[1]-1, bgr.shape[0]-1), bc, 4)
    ov = bgr.copy()
    cv2.rectangle(ov, (0,0), (bgr.shape[1], 36), (0,0,0), -1)
    cv2.addWeighted(ov, 0.5, bgr, 0.5, 0, bgr)
    cv2.putText(bgr, label, (6,18), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (255,255,255), 1, cv2.LINE_AA)
    if score is not None:
        cv2.putText(bgr, f"Score: {score:.3f}", (6,33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, bc, 1, cv2.LINE_AA)
    if conf is not None:
        cv2.putText(bgr, f"Conf: {conf:.1f}%", (6, bgr.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, bc, 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_normal_frames_sample(stem: str, max_frames: int = 80) -> list:
    frames_dir = cfg.FRAMES_DIR / stem
    if not frames_dir.exists():
        
        frames_dir = cfg.FRAMES_DIR
    if not frames_dir.exists():
        return []
    paths = sorted(frames_dir.glob("frame_*.jpg"))
    step  = max(1, len(paths) // max_frames)
    out   = []
    for i, p in enumerate(paths[::step][:max_frames]):
        img = cv2.imread(str(p))
        if img is not None:
            out.append((i * step, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return out


def model_ready() -> bool:
    return (
        (cfg.CHECKPOINTS_DIR / "temporal.pt").exists() and
        (cfg.CHECKPOINTS_DIR / "spatial.pt").exists()
    )


def calibration_ready() -> bool:
    return cfg.CALIBRATION_FILE.exists()





with st.sidebar:
    st.markdown("## 🔬 JEPA Anomaly Detector v7")
    st.markdown('<span class="phase-pill">5 Upgrades · Upload & Detect</span>',
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🎛 Status")
    normal_set = st.session_state["normal_video_path"] is not None
    st.markdown(f"{'✅' if normal_set   else '⭕'} Normal Video Uploaded")
    st.markdown(f"{'✅' if model_ready()          else '⭕'} Models Trained")
    st.markdown(f"{'✅' if cfg.ENERGY_FILE.exists() else '⭕'} Energy Model (SVDD)")
    st.markdown(f"{'✅' if calibration_ready()    else '⭕'} Calibrated")

    st.markdown("---")
    st.markdown("### ⚙ Config")
    with st.expander("Parameters"):
        fps        = st.slider("Sample FPS",          1, 10, cfg.TARGET_FPS)
        window_k   = st.slider("Window Size K (short)", 4, 16, cfg.WINDOW_SIZE)
        epochs     = st.slider("Epochs (JEPA)",       5, 100, cfg.EPOCHS, step=5)
        svdd_ep    = st.slider("Epochs (SVDD)",       5, 50,  cfg.SVDD_EPOCHS, step=5)
        batch_size = st.slider("Batch Size",          4, 64,  cfg.BATCH_SIZE, step=4)
        alpha      = st.slider("α Temporal-Short",    0.0, 1.0, cfg.SCORE_ALPHA,      step=0.05)
        alpha_long = st.slider("α Temporal-Long",     0.0, 1.0, cfg.SCORE_ALPHA_LONG, step=0.05)
        beta_v     = st.slider("β Spatial",           0.0, 1.0, cfg.SCORE_BETA,       step=0.05)
        gamma      = st.slider("γ Energy (SVDD)",     0.0, 1.0, cfg.SCORE_GAMMA,      step=0.05)

    st.markdown("### 🔦 Robustness")
    with st.expander("Lighting & Color"):
        cfg.GCN_ENABLE = st.toggle("Global Contrast Normalization", value=cfg.GCN_ENABLE, help="Reduces sensitivity to global lighting shifts.")
        cfg.USE_GRAYSCALE = st.toggle("Grayscale Mode", value=cfg.USE_GRAYSCALE, help="Ignores color shifts (useful if color doesn't matter for defects).")
        cfg.AUGMENT_LIGHTING = st.toggle("Lighting Augmentation (Training)", value=cfg.AUGMENT_LIGHTING, help="Trains model on varied lighting samples.")

    st.markdown(f"🖥 Device: **`{cfg.DEVICE.upper()}`**")
    st.markdown("---")
    st.caption("JEPA · ViT-B/16* · 5 Upgrades · 2026")





tab_train, tab_calib, tab_test = st.tabs([
    "🎓 1 · Train on Normal Video",
    "🎯 2 · Calibrate Threshold",
    "🔍 3 · Detect Anomalies",
])





with tab_train:
    st.markdown("## 🎓 Upload Normal Video & Train")
    st.markdown(
        "Upload a video that shows **normal, defect-free operation**. "
        "The model learns what 'normal' looks like — no labels needed.\n\n"
        "> **Training stages:** 3a — Short+Long Temporal + Cross-Attention Spatial JEPA  "
        "|  3b — Deep SVDD Energy Model"
    )
    st.markdown("---")

    
    st.markdown("### 📂 Step 1 — Upload Normal Video")
    uploaded_normal = st.file_uploader(
        "Choose a video file (MP4, AVI, MOV, MKV)",
        type=["mp4", "avi", "mov", "mkv"],
        key="normal_uploader",
    )

    if uploaded_normal:
        saved_path = save_uploaded_video(uploaded_normal)
        st.session_state["normal_video_path"] = saved_path
        st.session_state["normal_video_stem"] = saved_path.stem
        st.success(f"✅ Video saved: `{saved_path.name}`")
        st.video(str(saved_path))

    if st.session_state["normal_video_path"] is None:
        st.info("⬆ Upload a normal video above to begin.")
    else:
        st.markdown("---")
        st.markdown(f"### 🚀 Step 2 — Train on `{st.session_state['normal_video_stem']}`")

        start_training = st.button("🚀 Start Training (Stage 3a + 3b)", use_container_width=True)

        log_container   = st.empty()
        prog_container  = st.empty()
        chart_container = st.empty()

        if start_training:
            normal_path = st.session_state["normal_video_path"]
            log_lines, loss_hist = [], []

            def add_log(msg):
                log_lines.append(msg)
                log_container.markdown(
                    "**Log:**\n" + "\n".join(f"• {l}" for l in log_lines[-18:])
                )

            def on_progress(epoch, total, loss):
                loss_hist.append(loss)
                prog_container.progress(
                    epoch / total, text=f"Epoch {epoch}/{total} — Loss: {loss:.5f}"
                )
                if len(loss_hist) > 1:
                    chart_container.plotly_chart(
                        make_loss_curve(loss_hist),
                        key=f"train_live_{epoch}",
                    )

            try:
                from training.trainer import JEPATrainer
                cfg.TARGET_FPS  = fps
                cfg.WINDOW_SIZE = window_k

                trainer = JEPATrainer(
                    device=cfg.DEVICE,
                    log_callback=add_log,
                    progress_callback=on_progress,
                )

                
                add_log("📹 Extracting frames…")
                fp_bar = st.progress(0, "Sampling frames…")
                paths = trainer.extract_frames(
                    video_path=normal_path, fps=fps,
                    frame_progress=lambda c, t: fp_bar.progress(
                        min(c / max(t,1), 1.0), text=f"Frame {c}/{t}"
                    ),
                )
                fp_bar.empty()
                add_log(f"   → {len(paths)} frames extracted")

                
                add_log("🧠 Encoding with ViT-B/16* (partial domain adaptation)…")
                en_bar = st.progress(0, "Encoding…")
                trainer.encode_frames(
                    frame_paths=paths,
                    video_stem=normal_path.stem,
                    frame_progress=lambda c, t: en_bar.progress(
                        min(c / max(t,1), 1.0), text=f"Encoded {c}/{t}"
                    ),
                )
                en_bar.empty()

                
                add_log("🎯 Stage 3a — Short+Long Temporal + Cross-Attention Spatial JEPA…")
                loss_history = trainer.train(
                    epochs=epochs, batch_size=batch_size, lr=cfg.LEARNING_RATE
                )
                (cfg.CHECKPOINTS_DIR / "loss_history.json").write_text(json.dumps(loss_history))
                chart_container.plotly_chart(
                    make_loss_curve(loss_history), key="train_final"
                )

                
                add_log("⚡ Stage 3b — Training Deep SVDD energy model…")
                svdd_bar = st.progress(0, "SVDD training…")
                def on_svdd(ep, tot, loss):
                    svdd_bar.progress(ep / tot, text=f"SVDD {ep}/{tot} — {loss:.5f}")
                trainer.on_progress = on_svdd
                svdd_hist = trainer.train_energy(epochs=svdd_ep)
                svdd_bar.empty()
                add_log(f"✅ SVDD done — final loss: {svdd_hist[-1]:.5f}")

                st.session_state["trained"]     = True
                st.session_state["trainer_obj"] = trainer
                st.success(
                    f"✅ All training complete — JEPA: {loss_history[-1]:.5f} | "
                    f"SVDD: {svdd_hist[-1]:.5f}"
                )

            except Exception as e:
                st.error(f"Training failed: {e}")
                import traceback; st.code(traceback.format_exc())

        
        lf = cfg.CHECKPOINTS_DIR / "loss_history.json"
        if lf.exists() and not start_training:
            prior = json.loads(lf.read_text())
            st.markdown("**Previous Training Loss:**")
            st.plotly_chart(make_loss_curve(prior), key="train_prior")
            st.caption(f"Epochs: {len(prior)} | Final loss: {prior[-1]:.5f}")





with tab_calib:
    st.markdown("## 🎯 Calibrate Anomaly Threshold")
    st.markdown(
        "Runs all trained models on **normal video embeddings** to compute per-frame scores, "
        "then sets the threshold at the **97th percentile** and saves per-component "
        "normalisation scales."
    )

    if not model_ready():
        st.warning("⚠ Complete training in Tab 1 first.")
    else:
        calib_file = cfg.CALIBRATION_FILE
        if calib_file.exists():
            c = json.loads(calib_file.read_text())
            st.success(f"✅ Calibration done — Threshold: **{c['threshold']:.4f}**")
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Threshold",  f"{c['threshold']:.4f}")
            cc2.metric("t_scale",    f"{c.get('t_scale', 1.0):.4f}")
            cc3.metric("s_scale",    f"{c.get('s_scale', 1.0):.4f}")
            cc4.metric("e_scale",    f"{c.get('e_scale', 1.0):.4f}")
            st.session_state["threshold"] = c["threshold"]

        run_calib = st.button("🔄 Run Calibration", use_container_width=False)

        if run_calib:
            stem = st.session_state.get("normal_video_stem") or cfg.NORMAL_VIDEO.stem
            cls_path   = cfg.EMBEDDINGS_DIR / f"{stem}_cls.npy"
            patch_path = cfg.EMBEDDINGS_DIR / f"{stem}_patches.npy"

            if not cls_path.exists():
                st.error("Embeddings not found — run training first.")
            else:
                try:
                    from models.temporal_transformer import TemporalTransformer
                    from models.spatial_jepa import SpatialJEPAHead
                    from anomaly.energy_model import EnergyModel
                    from anomaly.scorer import AnomalyScorer
                    from training.calibration import calibrate_threshold, calibrate_component_scales

                    long_seq_len = max(cfg.LONG_WINDOW_SIZE // cfg.LONG_DOWNSAMPLE, 4)

                    clog = st.empty(); cbar = st.progress(0)

                    clog.info("Loading Short Temporal model…")
                    ts = TemporalTransformer(window_size=cfg.WINDOW_SIZE).to(cfg.DEVICE)
                    ts.load(cfg.CHECKPOINTS_DIR / "temporal.pt", cfg.DEVICE)

                    tl = TemporalTransformer(window_size=long_seq_len).to(cfg.DEVICE)
                    if (cfg.CHECKPOINTS_DIR / "temporal_long.pt").exists():
                        tl.load(cfg.CHECKPOINTS_DIR / "temporal_long.pt", cfg.DEVICE)

                    clog.info("Loading Cross-Attention Spatial JEPA…")
                    sm = SpatialJEPAHead().to(cfg.DEVICE)
                    sm.load(cfg.CHECKPOINTS_DIR / "spatial.pt", cfg.DEVICE)

                    clog.info("Loading PCA + Deep SVDD Energy model…")
                    pca_model = EnergyModel.load_pca()
                    if pca_model is not None:
                        pca_dim = pca_model.n_components_
                        em = EnergyModel(input_dim=pca_dim, proj_dim=cfg.SVDD_DIM).to(cfg.DEVICE)
                    else:
                        em = EnergyModel(input_dim=cfg.EMBED_DIM, proj_dim=cfg.SVDD_DIM).to(cfg.DEVICE)
                    if cfg.ENERGY_FILE.exists():
                        em.load(cfg.ENERGY_FILE, cfg.DEVICE)

                    clog.info("Loading embeddings…")
                    cls_embs   = np.load(str(cls_path))
                    patch_embs = np.load(str(patch_path))

                    
                    scorer_raw = AnomalyScorer(
                        temporal_short = ts,
                        temporal_long  = tl,
                        spatial_head   = sm,
                        energy_model   = em,
                        device         = cfg.DEVICE,
                        pca_model      = pca_model,
                        normal_cls_bank   = cls_embs,
                        normal_patch_bank = patch_embs,
                        
                    )

                    clog.info("Pass 1/2 — Measuring raw component magnitudes…")
                    t_errs, tl_errs, s_errs, e_errs, p_errs = [], [], [], [], []
                    N = len(cls_embs)
                    for i in range(N):
                        _, comp = scorer_raw.push_and_score(cls_embs[i], patch_embs[i])
                        if scorer_raw.is_ready():
                            t_errs.append(comp.get("temporal",      0.0))
                            tl_errs.append(comp.get("temporal_long", 0.0))
                            s_errs.append(comp.get("spatial",       0.0))
                            e_errs.append(comp.get("energy",        0.0))
                            p_errs.append(comp.get("patch_nn",      0.0))
                        cbar.progress((i+1)/(N*2), text=f"Pass 1 — Frame {i+1}/{N}")

                    scales = calibrate_component_scales(
                        np.array(t_errs), np.array(tl_errs),
                        np.array(s_errs), np.array(e_errs),
                        np.array(p_errs),
                    )

                    
                    clog.info("Pass 2/2 — Scoring with normalized scales…")
                    scorer_scaled = AnomalyScorer(
                        temporal_short = ts,
                        temporal_long  = tl,
                        spatial_head   = sm,
                        energy_model   = em,
                        device         = cfg.DEVICE,
                        t_scale        = scales["t_scale"],
                        t_long_scale   = scales["t_long_scale"],
                        s_scale        = scales["s_scale"],
                        e_scale        = scales["e_scale"],
                        p_scale        = scales["p_scale"],
                        pca_model      = pca_model,
                        normal_cls_bank   = cls_embs,
                        normal_patch_bank = patch_embs,
                    )

                    normal_scores = []
                    for i in range(N):
                        s, _ = scorer_scaled.push_and_score(cls_embs[i], patch_embs[i])
                        if scorer_scaled.is_ready():
                            normal_scores.append(s)
                        cbar.progress(0.5 + (i+1)/(N*2), text=f"Pass 2 — Frame {i+1}/{N}")

                    cbar.empty(); clog.empty()
                    normal_scores = np.array(normal_scores)

                    thr = calibrate_threshold(normal_scores, component_scales=scales)
                    st.session_state["threshold"] = thr
                    st.session_state["calibrated"] = True
                    
                    st.cache_resource.clear()

                    st.success(f"✅ Threshold set to **{thr:.4f}**")
                    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
                    cc1.metric("Threshold", f"{thr:.4f}")
                    cc2.metric("t_scale",   f"{scales['t_scale']:.4f}")
                    cc3.metric("s_scale",   f"{scales['s_scale']:.4f}")
                    cc4.metric("e_scale",   f"{scales['e_scale']:.4f}")
                    cc5.metric("p_scale",   f"{scales['p_scale']:.4f}")

                    fig = px.histogram(
                        x=normal_scores, nbins=50,
                        labels={"x": "Anomaly Score", "y": "Count"},
                        color_discrete_sequence=["#1a6ab5"],
                    )
                    fig.add_vline(x=thr, line_dash="dash", line_color="#e03030",
                                  annotation_text="Threshold",
                                  annotation_font_color="#e03030")
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f5f8ff",
                        font=dict(color="#3a5a8a"),
                        xaxis=dict(gridcolor="#dde8f5"), yaxis=dict(gridcolor="#dde8f5"),
                        margin=dict(l=10, r=10, t=30, b=10), height=240,
                        title="Normal Score Distribution", title_font_color="#2a4a7a",
                    )
                    st.plotly_chart(fig, key="calib_hist")

                except Exception as e:
                    st.error(f"Calibration failed: {e}")
                    import traceback; st.code(traceback.format_exc())





with tab_test:
    st.markdown("## 🔍 Detect Anomalies")

    if not model_ready() or not calibration_ready():
        st.warning("⚠ Complete training (Tab 1) and calibration (Tab 2) first.")
    else:
        from training.calibration import load_calibration
        calib_data = load_calibration()
        threshold  = calib_data["threshold"] if calib_data else st.session_state["threshold"]

        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown("### Choose Analysis Mode")
        with col_right:
            thr_mode = st.radio(
                "Threshold Mode", 
                ["Dynamic (Mean + N*Std)", "Dynamic (Mean * Multiplier)", "Dynamic (Test Video Mean)", "Use Calibrated", "Manual Threshold"],
                index=0
            )
            
            std_multiplier = 2.0
            mean_multiplier = 1.5
            
            if thr_mode in ["Use Calibrated", "Manual Threshold"]:
                threshold = st.number_input(
                    "Threshold Value", value=float(threshold),
                    min_value=0.001, step=0.001, format="%.4f",
                    key="thr_override",
                )
            elif thr_mode == "Dynamic (Mean + N*Std)":
                std_multiplier = st.number_input(
                    "Standard Deviation Buffer (N)", value=2.0, min_value=0.5, step=0.1, key="std_mult"
                )
            elif thr_mode == "Dynamic (Mean * Multiplier)":
                mean_multiplier = st.number_input(
                    "Percentage Buffer Multiplier", value=1.5, min_value=1.1, step=0.1, key="mean_mult"
                )

        mode = st.radio(
            "Analysis mode",
            ["📁 Upload Test Video", "📹 Live Webcam Stream"],
            horizontal=True, label_visibility="collapsed",
        )
        st.markdown("---")

        
        @st.cache_resource
        def get_scorer_and_encoder():
            from inference.pipeline import build_scorer
            stem = st.session_state.get("normal_video_stem", "") or cfg.NORMAL_VIDEO.stem
            scorer, encoder, _ = build_scorer(cfg.DEVICE, video_stem=stem)
            return scorer, encoder

        
        
        
        if mode == "📁 Upload Test Video":
            st.markdown("### 📂 Upload Test / Suspect Video")

            uploaded_test = st.file_uploader(
                "Choose test video", type=["mp4","avi","mov","mkv"],
                key="test_uploader",
            )

            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                max_frames_n = st.slider("Max frames (0 = all)", 0, 600, 0, step=10)
            with col_opt2:
                show_every = st.slider("Show every Nth frame", 1, 20, 5)

            run_test = st.button("▶ Run Detection", use_container_width=True,
                                  disabled=(uploaded_test is None))

            if run_test and uploaded_test:
                test_path = save_uploaded_video(uploaded_test)
                st.info(f"Analyzing `{test_path.name}` …")

                try:
                    scorer, encoder = get_scorer_and_encoder()
                    scorer.alpha      = alpha
                    scorer.alpha_long = alpha_long
                    scorer.beta       = beta_v
                    scorer.gamma      = gamma
                    scorer.reset()

                    from inference.pipeline import run_video_inference

                    cap_tmp = cv2.VideoCapture(str(test_path))
                    total_src  = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
                    src_fps    = cap_tmp.get(cv2.CAP_PROP_FPS) or 25.0
                    cap_tmp.release()
                    est_total = total_src // max(1, round(src_fps / cfg.TARGET_FPS))
                    max_f = max_frames_n if max_frames_n > 0 else None
                    if max_f: est_total = min(est_total, max_f)

                    total_prog  = st.progress(0, text="Starting…")
                    score_chart = st.empty()
                    frame_disp  = st.empty()
                    alert_disp  = st.empty()
                    results         = []
                    dynamic_scores  = []

                    def frame_cb(frame_rgb, result, fidx):
                        dynamic_scores.append(result["score"])
                        
                        if thr_mode == "Dynamic (Test Video Mean)":
                            current_thr = np.mean(dynamic_scores)
                        elif thr_mode == "Dynamic (Mean + N*Std)":
                            current_thr = np.mean(dynamic_scores) + std_multiplier * np.std(dynamic_scores) if len(dynamic_scores) > 1 else np.mean(dynamic_scores)
                        elif thr_mode == "Dynamic (Mean * Multiplier)":
                            current_thr = np.mean(dynamic_scores) * mean_multiplier
                        else:
                            current_thr = threshold

                        result["is_anomaly"] = float(result["score"]) > float(current_thr)
                        result["confidence"] = calculate_confidence(result.get("uncertainty", 0.0))
                        results.append(result)
                        
                        pct = min((fidx+1) / max(est_total,1), 1.0)
                        total_prog.progress(
                            pct, text=f"Frame {fidx+1}/{est_total} — Score: {result['score']:.4f} | Conf: {result['confidence']:.1f}% | Thr: {current_thr:.3f}"
                        )
                        if len(results) % 5 == 0 and len(results) > 1:
                            score_chart.plotly_chart(
                                make_score_timeline(results, float(current_thr)),
                                key=f"tl_live_{len(results)}",
                            )
                        if result["is_anomaly"]:
                            alert_disp.markdown(
                                '<div class="alert-anomaly">🚨 ANOMALY DETECTED</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            alert_disp.markdown(
                                '<div class="alert-normal">✅ Normal</div>',
                                unsafe_allow_html=True
                            )
                        if fidx % show_every == 0:
                            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                            clr = (0,180,80) if not result["is_anomaly"] else (0,60,255)
                            cv2.putText(bgr, f"Score: {result['score']:.3f}",
                                        (8,24), cv2.FONT_HERSHEY_SIMPLEX, .7, clr, 2)
                            cv2.putText(bgr, f"Conf: {result['confidence']:.1f}%",
                                        (8, bgr.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, .7, clr, 2)
                            if result["is_anomaly"]:
                                cv2.rectangle(bgr,(0,0),(bgr.shape[1]-1,bgr.shape[0]-1),
                                              (0,60,255),4)
                            frame_disp.image(
                                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                caption=f"Frame {fidx}",
                                use_container_width=True,
                            )

                    run_video_inference(
                        test_path, scorer, encoder, 
                        threshold if thr_mode == "Use Calibrated" else 999.0, 
                        frame_callback=frame_cb, max_frames=max_f,
                    )
                    
                    
                    if "Dynamic" in thr_mode and len(dynamic_scores) > 0:
                        final_mean = np.mean(dynamic_scores)
                        final_std  = np.std(dynamic_scores)
                        
                        if thr_mode == "Dynamic (Test Video Mean)":
                            final_thr = final_mean
                        elif thr_mode == "Dynamic (Mean + N*Std)":
                            final_thr = final_mean + std_multiplier * final_std
                        elif thr_mode == "Dynamic (Mean * Multiplier)":
                            final_thr = final_mean * mean_multiplier
                            
                        
                        for r in results:
                            r["is_anomaly"] = float(r["score"]) > float(final_thr)
                            # Confidence is unchanged as it relies on uncertainty, not threshold
                            
                        
                        threshold = float(final_thr)
                        
                    total_prog.progress(1.0, text=f"✅ Complete! (Final Threshold: {threshold:.3f})")

                    
                    if results:
                        st.markdown("---")
                        st.markdown("### 📊 Summary")
                        n_tot  = len(results)
                        n_anom = sum(1 for r in results if r["is_anomaly"])
                        c1,c2,c3,c4 = st.columns(4)
                        c1.metric("Total Frames",   n_tot)
                        c2.metric("Anomaly Frames", n_anom,
                                  delta=f"{n_anom/n_tot*100:.1f}% flagged",
                                  delta_color="inverse")
                        c3.metric("Max Score",  f"{max(r['score'] for r in results):.4f}")
                        c4.metric("Avg Conf", f"{np.mean([r.get('confidence', 0.0) for r in results]):.1f}%")

                        st.plotly_chart(
                            make_score_timeline(results, threshold), key="tl_final"
                        )

                        
                        last_r = results[-1]
                        st.markdown("#### Last Frame Score Breakdown")
                        gc, bc_col = st.columns(2)
                        with gc:
                            st.plotly_chart(make_gauge(last_r["score"], threshold),
                                            key="gauge_up")
                        with bc_col:
                            st.plotly_chart(make_component_bar(last_r), key="bar_up")

                        st.markdown(
                            "*Component legend: T-Short = Temporal (K=8, uncertainty-damped) · "
                            "T-Long = Temporal (K=32 drift) · Spatial = Cross-attn patches · "
                            "Energy = Deep SVDD · Uncertainty = MC dropout variance*"
                        )

                        
                        anomaly_frames = [r for r in results if r["is_anomaly"]]
                        if anomaly_frames:
                            st.markdown(f"#### 🚨 {len(anomaly_frames)} Anomaly Frame(s)")
                            for r in anomaly_frames[:20]:
                                st.write(
                                    f"• Frame **{r['frame_idx']}** — "
                                    f"Score: `{r['score']:.4f}` "
                                    f"(T-S: {r['temporal']:.3f}, "
                                    f"T-L: {r.get('temporal_long',0):.3f}, "
                                    f"Sp: {r['spatial']:.3f}, "
                                    f"E: {r.get('energy',0):.3f}, "
                                    f"Unc: {r.get('uncertainty',0):.4f})"
                                )
                            if len(anomaly_frames) > 20:
                                st.caption(f"… and {len(anomaly_frames) - 20} more")

                            
                            st.markdown("---")
                            st.markdown("### 🖼 Normal vs Anomaly — Side-by-Side (Object-Level Semantic Match)")
                            stem = st.session_state.get("normal_video_stem") or cfg.NORMAL_VIDEO.stem
                            normal_sample = load_normal_frames_sample(stem)
                            top_anom = sorted(anomaly_frames,
                                              key=lambda r: r["score"], reverse=True)[:10]
                            if normal_sample:
                                
                                normal_embs = []
                                normal_patches = []
                                for _, n_fr in normal_sample:
                                    n_rgb = cv2.resize(n_fr, (cfg.FRAME_SIZE, cfg.FRAME_SIZE))
                                    n_cls, n_patch = encoder.encode_frame_np(n_rgb)
                                    normal_embs.append(n_cls)
                                    normal_patches.append(n_patch)

                                try:
                                    from fpdf import FPDF
                                    pdf = FPDF()
                                    pdf.set_auto_page_break(auto=True, margin=15)
                                    pdf_tmp_dir = tempfile.mkdtemp()
                                    has_fpdf = True
                                except ImportError:
                                    has_fpdf = False

                                pdf_prog = st.progress(0, text="Generating side-by-side matches (and PDF Report)...")

                                
                                all_anom_sorted = sorted(anomaly_frames, key=lambda r: r["score"], reverse=True)

                                for pi, ar in enumerate(all_anom_sorted):
                                    a_rgb = cv2.resize(ar["frame_rgb"], (cfg.FRAME_SIZE, cfg.FRAME_SIZE))
                                    a_cls, a_patch = encoder.encode_frame_np(a_rgb)
                                    
                                    
                                    
                                    sims = [np.dot(a_cls.flatten(), n_cls.flatten()) / 
                                            (np.linalg.norm(a_cls)*np.linalg.norm(n_cls) + 1e-8) 
                                            for n_cls in normal_embs]
                                    best_ni = int(np.argmax(sims))
                                    
                                    norm_idx, norm_fr = normal_sample[best_ni]
                                    best_n_patch = normal_patches[best_ni]
                                    
                                    
                                    
                                    patch_diff = np.linalg.norm(a_patch - best_n_patch, axis=-1)  
                                    
                                    
                                    diff_min, diff_max = patch_diff.min(), patch_diff.max()
                                    if diff_max > diff_min:
                                        patch_diff = (patch_diff - diff_min) / (diff_max - diff_min)
                                    else:
                                        patch_diff = np.zeros_like(patch_diff)
                                        
                                    
                                    heatmap_14x14 = patch_diff.reshape(14, 14)
                                    
                                    
                                    heatmap_resized = cv2.resize(heatmap_14x14, (ar["frame_rgb"].shape[1], ar["frame_rgb"].shape[0]))
                                    
                                    
                                    heatmap_color = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                                    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                                    
                                    
                                    overlay_img = cv2.addWeighted(ar["frame_rgb"], 0.6, heatmap_rgb, 0.4, 0)
                                    
                                    na = annotate_frame(norm_fr, f"NORMAL #{norm_idx}", is_anomaly=False)
                                    aa = annotate_frame(ar["frame_rgb"],
                                                        f"ANOMALY #{ar['frame_idx']}",
                                                        score=ar["score"], conf=ar.get("confidence", 0.0), is_anomaly=True)
                                    
                                    
                                    if pi < 10:
                                        col_n, col_a, col_h = st.columns(3)
                                        with col_n:
                                            st.image(na, caption=f"✅ Normal · frame {norm_idx}",
                                                     use_container_width=True)
                                        with col_a:
                                            st.image(aa,
                                                     caption=f"🚨 Anomaly · frame {ar['frame_idx']} · {ar['score']:.3f} · Conf: {ar.get('confidence', 0.0):.1f}%",
                                                     use_container_width=True)
                                        with col_h:
                                            st.image(overlay_img,
                                                     caption=f"🔥 Anomaly Heatmap (Patch Diff)",
                                                     use_container_width=True)
                                        st.markdown(
                                            "<hr style='border:0;border-top:1px solid #dde8f5;margin:4px 0'>",
                                            unsafe_allow_html=True)
                                            
                                    if has_fpdf:
                                        import os
                                        norm_path = os.path.join(pdf_tmp_dir, f"n_{pi}.jpg")
                                        anom_path = os.path.join(pdf_tmp_dir, f"a_{pi}.jpg")
                                        heat_path = os.path.join(pdf_tmp_dir, f"h_{pi}.jpg")
                                        cv2.imwrite(norm_path, cv2.cvtColor(na, cv2.COLOR_RGB2BGR))
                                        cv2.imwrite(anom_path, cv2.cvtColor(aa, cv2.COLOR_RGB2BGR))
                                        cv2.imwrite(heat_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
                                        
                                        pdf.add_page()
                                        pdf.set_font("helvetica", 'B', 16)
                                        pdf.cell(0, 10, txt=f"Anomaly at Frame {ar['frame_idx']} (Score: {ar['score']:.3f} | Conf: {ar.get('confidence', 0.0):.1f}%)", ln=True, align='C')
                                        pdf.ln(10)
                                        
                                        pdf.image(norm_path, x=10, y=40, w=60)
                                        pdf.image(anom_path, x=75, y=40, w=60)
                                        pdf.image(heat_path, x=140, y=40, w=60)
                                        
                                        pdf.set_font("helvetica", size=12)
                                        pdf.set_xy(10, 90)
                                        pdf.cell(60, 10, txt=f"Normal (Frame {norm_idx})", ln=0, align='C')
                                        pdf.set_xy(75, 90)
                                        pdf.cell(60, 10, txt="Anomaly Frame", ln=0, align='C')
                                        pdf.set_xy(140, 90)
                                        pdf.cell(60, 10, txt="Heatmap", ln=1, align='C')

                                        
                                        pdf.ln(10)
                                        pdf.set_font("helvetica", 'B', 13)
                                        pdf.set_text_color(26, 106, 181) 
                                        pdf.cell(0, 10, txt="Component Score Breakdown", ln=True)
                                        pdf.set_text_color(0, 0, 0)
                                        pdf.set_font("helvetica", size=11)
                                        
                                        
                                        pdf.set_font("helvetica", 'B', 11)
                                        pdf.cell(80, 8, txt="Metric", border=1, ln=0, align='C')
                                        pdf.cell(40, 8, txt="Score", border=1, ln=1, align='C')
                                        
                                        pdf.set_font("helvetica", size=11)
                                        metrics = [
                                            ("Confidence (MC Dropout)", f"{ar.get('confidence', 0.0):.1f}%"),
                                            ("Spatial (Reconstruction)", f"{ar['spatial']:.4f}"),
                                            ("Temporal (Short Prediction)", f"{ar['temporal']:.4f}"),
                                            ("Temporal (Long Drift)", f"{ar.get('temporal_long', 0):.4f}"),
                                            ("Deep SVDD (Energy)", f"{ar.get('energy', 0):.4f}"),
                                            ("Patch Nearest-Neighbor", f"{ar.get('patch_nn', 0):.4f}"),
                                            ("Uncertainty (MC Dropout)", f"{ar.get('uncertainty', 0):.4f}")
                                        ]
                                        
                                        for label, val in metrics:
                                            pdf.cell(80, 8, txt=label, border=1, ln=0)
                                            pdf.cell(40, 8, txt=val, border=1, ln=1, align='C')

                                    pdf_prog.progress(min((pi+1)/len(all_anom_sorted), 1.0), text=f"Processed frame {pi+1}/{len(all_anom_sorted)}")

                                pdf_prog.empty()
                                st.caption(f"Top {min(10, len(all_anom_sorted))} anomaly frames rendered above.")
                                
                                if has_fpdf:
                                    pdf_path = os.path.join(pdf_tmp_dir, "anomaly_report.pdf")
                                    pdf.output(pdf_path)
                                    with open(pdf_path, "rb") as f:
                                        pdf_bytes = f.read()
                                    st.download_button(
                                        label="📄 Download Full PDF Report (All Anomalies)",
                                        data=pdf_bytes,
                                        file_name="JEPA_Anomaly_Report.pdf",
                                        mime="application/pdf",
                                        use_container_width=True
                                    )
                                else:
                                    st.warning("Install `fpdf2` to enable PDF report downloading.")
                            else:
                                st.warning("⚠ No cached training frames found. Re-run training first.")
                        else:
                            st.success("✅ No anomalies detected — video looks normal.")

                except Exception as e:
                    st.error(f"Inference failed: {e}")
                    import traceback; st.code(traceback.format_exc())

        
        
        
        else:
            st.markdown("### 📹 Live Webcam Stream")
            st.markdown(
                "The webcam feed is analyzed frame-by-frame. "
                "Each frame is scored using all 5 model components."
            )

            col_l, col_r = st.columns(2)
            with col_l:
                cam_index = st.number_input("Camera index (0 = default)", 0, 5, 0, step=1)
            with col_r:
                analyze_every = st.slider("Analyze every Nth frame", 1, 10, 2)

            start_stream = st.button("▶ Start Stream", use_container_width=True)
            stop_holder  = st.empty()

            live_frame_disp  = st.empty()
            live_alert_disp  = st.empty()
            live_score_disp  = st.empty()
            live_chart_disp  = st.empty()
            live_gauge_disp  = st.empty()

            if start_stream:
                try:
                    scorer, encoder = get_scorer_and_encoder()
                    scorer.alpha      = alpha
                    scorer.alpha_long = alpha_long
                    scorer.beta       = beta_v
                    scorer.gamma      = gamma
                    scorer.reset()

                    cap = cv2.VideoCapture(int(cam_index))
                    if not cap.isOpened():
                        st.error(f"Cannot open camera {cam_index}.")
                    else:
                        st.session_state["webcam_running"] = True
                        stop_btn = stop_holder.button("⏹ Stop Stream", key="stop_webcam")

                        frame_count    = 0
                        webcam_results = []
                        webcam_scores  = []

                        while st.session_state["webcam_running"]:
                            ret, frame_bgr = cap.read()
                            if not ret:
                                st.warning("Webcam feed ended.")
                                break

                            frame_resized = cv2.resize(frame_bgr, (cfg.FRAME_SIZE, cfg.FRAME_SIZE))
                            frame_count += 1

                            if frame_count % analyze_every == 0:
                                
                                lab = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2LAB)
                                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                                frame_norm = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                                frame_rgb = cv2.cvtColor(frame_norm, cv2.COLOR_BGR2RGB)

                                cls_emb, patch_emb = encoder.encode_frame_np(frame_rgb)
                                score, components  = scorer.push_and_score(cls_emb, patch_emb)
                                
                                webcam_scores.append(score)
                                if thr_mode == "Dynamic (Test Video Mean)":
                                    current_thr = np.mean(webcam_scores)
                                elif thr_mode == "Dynamic (Mean + N*Std)":
                                    current_thr = np.mean(webcam_scores) + std_multiplier * np.std(webcam_scores) if len(webcam_scores) > 1 else np.mean(webcam_scores)
                                elif thr_mode == "Dynamic (Mean * Multiplier)":
                                    current_thr = np.mean(webcam_scores) * mean_multiplier
                                else:
                                    current_thr = threshold

                                is_anomaly = float(score) > float(current_thr)
                                conf = calculate_confidence(components.get("uncertainty", 0.0))

                                webcam_results.append({
                                    "frame_idx":   frame_count,
                                    "score":       score,
                                    "confidence":  conf,
                                    "is_anomaly":  is_anomaly,
                                    **components,
                                })

                                
                                disp_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                                clr = (0, 60, 255) if is_anomaly else (0, 180, 80)
                                cv2.putText(disp_bgr, f"Score: {score:.3f}",
                                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .7, clr, 2)
                                cv2.putText(disp_bgr, f"Conf: {conf:.1f}%",
                                            (8, disp_bgr.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, .7, clr, 2)
                                cv2.putText(disp_bgr,
                                            f"E:{components.get('energy',0):.3f}  "
                                            f"Unc:{components.get('uncertainty',0):.4f}",
                                            (8, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, clr, 1)
                                if is_anomaly:
                                    cv2.rectangle(disp_bgr,(0,0),
                                                  (disp_bgr.shape[1]-1, disp_bgr.shape[0]-1),
                                                  (0,60,255), 4)

                                live_frame_disp.image(
                                    cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB),
                                    caption=f"Frame {frame_count} | Score {score:.3f}",
                                    use_container_width=True,
                                )
                                if is_anomaly:
                                    live_alert_disp.markdown(
                                        '<div class="alert-anomaly">🚨 ANOMALY DETECTED</div>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    live_alert_disp.markdown(
                                        '<div class="alert-normal">✅ Normal Operation</div>',
                                        unsafe_allow_html=True
                                    )
                                live_score_disp.metric(
                                    "Composite Score", f"{score:.4f}",
                                    delta=f"threshold={current_thr:.3f}"
                                )
                                if len(webcam_results) > 2:
                                    live_chart_disp.plotly_chart(
                                        make_score_timeline(webcam_results[-100:], float(current_thr)),
                                        key=f"wcam_{frame_count}",
                                    )
                                    live_gauge_disp.plotly_chart(
                                        make_gauge(score, float(current_thr)),
                                        key=f"wcam_gauge_{frame_count}",
                                    )

                            if stop_btn:
                                st.session_state["webcam_running"] = False
                                break
                            time.sleep(0.03)

                        cap.release()
                        st.session_state["webcam_running"] = False
                        st.info("📹 Stream stopped.")

                        if webcam_results:
                            n_anom = sum(1 for r in webcam_results if r["is_anomaly"])
                            st.metric("Total Frames Analyzed", len(webcam_results))
                            st.metric("Anomaly Frames",        n_anom)

                except Exception as e:
                    st.error(f"Webcam error: {e}")
                    import traceback; st.code(traceback.format_exc())



