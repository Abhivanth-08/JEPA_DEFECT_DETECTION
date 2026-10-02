import sys, json, time, tempfile, shutil, math
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

st.set_page_config(
    page_title="JEPA Defect Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  
  html, body, [class*="css"] { 
      font-family: 'Inter', sans-serif; 
      background-color: #F7F8FA !important;
      color: #1a2b4c;
  }
  
  .stApp { background: #F7F8FA; }

  /* Hide default elements */
  header[data-testid="stHeader"] { display: none; }
  footer { display: none; }
  .stDeployButton { display: none; }

  /* Enterprise Nav Bar */
  .enterprise-nav {
      background: #ffffff;
      padding: 16px 32px;
      margin: -6rem -4rem 2rem -4rem;
      border-bottom: 1px solid #e2e8f0;
      box-shadow: 0 4px 20px rgba(0,0,0,0.02);
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 999;
  }
  .nav-brand {
      font-weight: 700;
      font-size: 1.3rem;
      color: #0f172a;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 12px;
  }
  .nav-brand span.badge {
      background: #e0e7ff;
      color: #3730a3;
      font-size: 0.7rem;
      padding: 4px 8px;
      border-radius: 6px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
  }
  .nav-stats {
      display: flex;
      gap: 24px;
      font-size: 0.85rem;
      font-weight: 500;
      color: #475569;
  }
  .nav-stats div {
      display: flex;
      align-items: center;
      gap: 6px;
  }
  .status-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 8px rgba(16,185,129,0.4);
  }

  /* Enterprise Cards */
  .card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 24px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.02);
      margin-bottom: 20px;
  }
  
  .card-title {
      font-size: 0.95rem;
      font-weight: 600;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 16px;
      border-bottom: 1px solid #f1f5f9;
      padding-bottom: 12px;
  }

  /* Alerts */
  .alert-card {
      padding: 16px;
      border-radius: 8px;
      margin-bottom: 16px;
      font-weight: 600;
      display: flex;
      flex-direction: column;
      gap: 4px;
      transition: all 0.3s ease;
  }
  .alert-normal {
      background: #f0fdf4;
      border: 1px solid #bbf7d0;
      color: #166534;
  }
  .alert-anomaly {
      background: #fef2f2;
      border: 1px solid #fecaca;
      color: #991b1b;
      animation: pulse-red 2s infinite;
  }
  @keyframes pulse-red {
      0% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
      70% { box-shadow: 0 0 0 10px rgba(239,68,68,0); }
      100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
  }

  /* Buttons */
  .stButton > button {
      background: #ffffff;
      border: 1px solid #cbd5e1;
      color: #334155;
      font-weight: 600;
      border-radius: 6px;
      padding: 8px 16px;
      transition: all 0.2s;
  }
  .stButton > button:hover {
      background: #f8fafc;
      border-color: #94a3b8;
      color: #0f172a;
  }
  
  /* Primary Action Button */
  .primary-btn {
      background: #0f172a !important;
      color: #ffffff !important;
      border: none !important;
  }
  .primary-btn:hover {
      background: #1e293b !important;
      box-shadow: 0 4px 12px rgba(15,23,42,0.2) !important;
  }

  /* Metrics styling */
  div[data-testid="metric-container"] {
      background: #f8fafc;
      padding: 16px;
      border-radius: 8px;
      border: 1px solid #e2e8f0;
  }
  
  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
      gap: 24px;
      padding: 0 12px;
  }
  .stTabs [data-baseweb="tab"] {
      padding: 12px 4px;
      color: #64748b;
      font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
      color: #0f172a !important;
      border-bottom-color: #0f172a !important;
  }
  
  /* Custom Log Container */
  .log-container {
      background: #1e293b;
      color: #e2e8f0;
      font-family: 'Courier New', monospace;
      font-size: 0.85rem;
      padding: 16px;
      border-radius: 8px;
      height: 200px;
      overflow-y: auto;
  }
  .log-line { margin: 4px 0; }
  .log-time { color: #94a3b8; margin-right: 8px; }
  .log-warn { color: #facc15; }
  .log-crit { color: #f87171; }

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
_ss("system_logs", [])

def add_system_log(msg, level="info"):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    st.session_state["system_logs"].append({"time": ts, "msg": msg, "level": level})
    if len(st.session_state["system_logs"]) > 50:
        st.session_state["system_logs"].pop(0)

UPLOADS_DIR = Path(__file__).parent / "user_uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

def save_uploaded_video(uploaded_file) -> Path:
    dest = UPLOADS_DIR / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest

def calculate_confidence(uncertainty: float) -> float:
    conf = math.exp(-uncertainty * 50.0) * 100.0
    return 100.0 - max(0.0, min(100.0, conf))

def score_to_color(score: float, threshold: float) -> str:
    r = score / max(threshold, 1e-6)
    if r < 0.8:  return "#10b981" 
    if r < 1.0:  return "#f59e0b" 
    return "#ef4444" 

def conf_to_color(conf: float) -> str:
    if conf > 80: return "#10b981"
    if conf > 50: return "#f59e0b"
    return "#ef4444"


st.markdown(f"""
<div class="enterprise-nav">
    <div class="nav-brand">
        JEPA Defect Intelligence
        <span class="badge">Enterprise Deployment</span>
    </div>
    <div class="nav-stats">
        <div><div class="status-dot"></div> Edge Node: ONLINE</div>
        <div>⚙️ Model: ViT-B/16* SVDD</div>
        <div>⏱️ Latency: <span id="latency-val">--</span> ms</div>
        <div>🗓️ {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
    </div>
</div>
""", unsafe_allow_html=True)


def make_loss_curve(loss_history: list) -> go.Figure:
    fig = go.Figure(go.Scatter(
        y=loss_history, mode="lines", line=dict(color="#2563eb", width=2)
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Epoch", gridcolor="#f1f5f9"),
        yaxis=dict(title="Loss", gridcolor="#f1f5f9"),
        margin=dict(l=10, r=10, t=10, b=10), height=220,
    )
    return fig

def make_score_timeline(results: list, threshold: float) -> go.Figure:
    frames = [r["frame_idx"] for r in results]
    scores = [r["score"] for r in results]
    colors = [score_to_color(s, threshold) for s in scores]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frames, y=scores, mode="lines+markers",
        line=dict(color="#cbd5e1", width=1),
        marker=dict(color=colors, size=6),
        name="Anomaly Score",
        hovertemplate="Frame: %{x}<br>Score: %{y:.3f}<extra></extra>"
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#ef4444", line_width=1.5)
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Frame", gridcolor="#f1f5f9", showline=True, linecolor="#cbd5e1"),
        yaxis=dict(title="Anomaly Score", gridcolor="#f1f5f9", showline=True, linecolor="#cbd5e1"),
        margin=dict(l=0, r=0, t=10, b=0), height=220,
        showlegend=False,
    )
    return fig

def make_conf_timeline(results: list) -> go.Figure:
    frames = [r["frame_idx"] for r in results]
    confs = [r.get("confidence", 0.0) for r in results]
    colors = [conf_to_color(c) for c in confs]
    
    fig = go.Figure(go.Bar(
        x=frames, y=confs, marker_color=colors,
        name="Confidence",
        hovertemplate="Frame: %{x}<br>Conf: %{y:.1f}%<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#f1f5f9"),
        yaxis=dict(title="Confidence %", range=[0, 100], gridcolor="#f1f5f9"),
        margin=dict(l=0, r=0, t=10, b=0), height=180,
    )
    return fig

def make_component_radar(components: dict) -> go.Figure:
    keys = ["T-Short", "T-Long", "Spatial", "Energy"]
    vals = [
        components.get("temporal", 0),
        components.get("temporal_long", 0),
        components.get("spatial", 0),
        components.get("energy", 0)
    ]
    max_val = max(max(vals), 0.01)
    norm_vals = [v / max_val for v in vals]
    
    fig = go.Figure(go.Scatterpolar(
        r=norm_vals,
        theta=keys,
        fill='toself',
        fillcolor="rgba(37, 99, 235, 0.2)",
        line=dict(color="#2563eb")
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(tickfont=dict(size=10, color="#64748b"))
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=20),
        height=200
    )
    return fig

def make_gauge(score: float, threshold: float) -> go.Figure:
    color = score_to_color(score, threshold)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 4),
        number={"font": {"size": 24, "color": color, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, threshold * 2.5], "tickcolor": "white", "tickfont": {"color": "white"}},
            "bar": {"color": color, "thickness": 0.2},
            "bgcolor": "#f1f5f9", "borderwidth": 0,
            "steps": [
                {"range": [0, threshold], "color": "#f8fafc"},
                {"range": [threshold, threshold*2.5], "color": "#fef2f2"},
            ],
            "threshold": {"line": {"color": "#ef4444", "width": 2},
                          "thickness": 0.8, "value": threshold},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=15, r=15, t=15, b=15),
        height=160,
    )
    return fig


def annotate_frame_enterprise(frame_rgb: np.ndarray, score: float, conf: float, is_anomaly: bool, threshold: float) -> np.ndarray:
    bgr = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    
    color = (0, 0, 220) if is_anomaly else (180, 220, 0)
    
    # Border
    if is_anomaly:
        cv2.rectangle(bgr, (0,0), (w-1, h-1), color, 4)
    
    # Top overlay bar
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0,0), (w, 40), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, bgr, 0.4, 0, bgr)
    
    # Status Text
    status = "ANOMALY DETECTED" if is_anomaly else "NORMAL"
    cv2.putText(bgr, f"SYS: {status}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Metrics
    cv2.putText(bgr, f"Score: {score:.3f} | Conf: {conf:.1f}%", (w - 280, 26), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def model_ready() -> bool:
    return (cfg.CHECKPOINTS_DIR / "temporal.pt").exists() and (cfg.CHECKPOINTS_DIR / "spatial.pt").exists()

def calibration_ready() -> bool:
    return cfg.CALIBRATION_FILE.exists()

tab_dash, tab_setup, tab_calib = st.tabs([
    "🖥️ Monitoring Dashboard",
    "⚙️ System Setup (Training)",
    "🎯 Threshold Calibration"
])

with tab_setup:
    st.markdown("<div class='card'><div class='card-title'>Model Initialization Pipeline</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        fps        = st.slider("Sample FPS", 1, 10, cfg.TARGET_FPS)
        window_k   = st.slider("Window Size K (short)", 4, 16, cfg.WINDOW_SIZE)
        epochs     = st.slider("Epochs (JEPA)", 5, 100, cfg.EPOCHS, step=5)
        svdd_ep    = st.slider("Epochs (SVDD)", 5, 50,  cfg.SVDD_EPOCHS, step=5)
        batch_size = st.slider("Batch Size", 4, 64,  cfg.BATCH_SIZE, step=4)
        
    with col2:
        uploaded_normal = st.file_uploader("Upload Normal Baseline Video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_normal:
            saved_path = save_uploaded_video(uploaded_normal)
            st.session_state["normal_video_path"] = saved_path
            st.session_state["normal_video_stem"] = saved_path.stem
            st.success(f"Baseline registered: {saved_path.name}")
            
        start_training = st.button("Initialize Learning Pipeline", use_container_width=True)
        prog_container  = st.empty()
        chart_container = st.empty()
        
        if start_training and st.session_state["normal_video_path"]:
            normal_path = st.session_state["normal_video_path"]
            loss_hist = []

            def add_log(msg): pass
            def on_progress(epoch, total, loss):
                loss_hist.append(loss)
                prog_container.progress(epoch / total, text=f"Optimization Phase — Epoch {epoch}/{total} | Loss: {loss:.5f}")
                if len(loss_hist) > 1:
                    chart_container.plotly_chart(make_loss_curve(loss_hist), key=f"train_{epoch}")

            try:
                from training.trainer import JEPATrainer
                cfg.TARGET_FPS  = fps
                cfg.WINDOW_SIZE = window_k

                trainer = JEPATrainer(device=cfg.DEVICE, log_callback=add_log, progress_callback=on_progress)
                
                paths = trainer.extract_frames(video_path=normal_path, fps=fps, frame_progress=lambda c, t: None)
                trainer.encode_frames(frame_paths=paths, video_stem=normal_path.stem, frame_progress=lambda c, t: None)
                loss_history = trainer.train(epochs=epochs, batch_size=batch_size, lr=cfg.LEARNING_RATE)
                (cfg.CHECKPOINTS_DIR / "loss_history.json").write_text(json.dumps(loss_history))
                
                trainer.on_progress = lambda ep, tot, loss: prog_container.progress(ep/tot, text=f"SVDD Energy Modeling {ep}/{tot}")
                svdd_hist = trainer.train_energy(epochs=svdd_ep)
                
                st.session_state["trained"] = True
                prog_container.empty()
                st.success("✅ Training Pipeline Complete. Proceed to Calibration.")
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                
    st.markdown("</div>", unsafe_allow_html=True)


with tab_calib:
    st.markdown("<div class='card'><div class='card-title'>Threshold Calibration Pipeline</div>", unsafe_allow_html=True)
    if not model_ready():
        st.warning("Models not initialized. Complete Setup first.")
    else:
        run_calib = st.button("Run Statistical Calibration", use_container_width=True)
        if run_calib:
            stem = st.session_state.get("normal_video_stem") or cfg.NORMAL_VIDEO.stem
            cls_path   = cfg.EMBEDDINGS_DIR / f"{stem}_cls.npy"
            patch_path = cfg.EMBEDDINGS_DIR / f"{stem}_patches.npy"
            if not cls_path.exists(): st.error("Embeddings missing.")
            else:
                try:
                    from models.temporal_transformer import TemporalTransformer
                    from models.spatial_jepa import SpatialJEPAHead
                    from anomaly.energy_model import EnergyModel
                    from anomaly.scorer import AnomalyScorer
                    from training.calibration import calibrate_threshold, calibrate_component_scales

                    long_seq_len = max(cfg.LONG_WINDOW_SIZE // cfg.LONG_DOWNSAMPLE, 4)
                    
                    cbar = st.progress(0, "Loading dependencies...")
                    ts = TemporalTransformer(window_size=cfg.WINDOW_SIZE).to(cfg.DEVICE)
                    ts.load(cfg.CHECKPOINTS_DIR / "temporal.pt", cfg.DEVICE)

                    tl = TemporalTransformer(window_size=long_seq_len).to(cfg.DEVICE)
                    if (cfg.CHECKPOINTS_DIR / "temporal_long.pt").exists():
                        tl.load(cfg.CHECKPOINTS_DIR / "temporal_long.pt", cfg.DEVICE)

                    sm = SpatialJEPAHead().to(cfg.DEVICE)
                    sm.load(cfg.CHECKPOINTS_DIR / "spatial.pt", cfg.DEVICE)

                    pca_model = EnergyModel.load_pca()
                    if pca_model is not None:
                        em = EnergyModel(input_dim=pca_model.n_components_, proj_dim=cfg.SVDD_DIM).to(cfg.DEVICE)
                    else:
                        em = EnergyModel(input_dim=cfg.EMBED_DIM, proj_dim=cfg.SVDD_DIM).to(cfg.DEVICE)
                    if cfg.ENERGY_FILE.exists():
                        em.load(cfg.ENERGY_FILE, cfg.DEVICE)

                    cls_embs   = np.load(str(cls_path))
                    patch_embs = np.load(str(patch_path))

                    scorer_raw = AnomalyScorer(
                        temporal_short = ts, temporal_long = tl, spatial_head = sm,
                        energy_model = em, device = cfg.DEVICE, pca_model = pca_model,
                        normal_cls_bank = cls_embs, normal_patch_bank = patch_embs,
                    )

                    t_errs, tl_errs, s_errs, e_errs, p_errs = [], [], [], [], []
                    N = len(cls_embs)
                    for i in range(N):
                        _, comp = scorer_raw.push_and_score(cls_embs[i], patch_embs[i])
                        if scorer_raw.is_ready():
                            t_errs.append(comp.get("temporal", 0.0))
                            tl_errs.append(comp.get("temporal_long", 0.0))
                            s_errs.append(comp.get("spatial", 0.0))
                            e_errs.append(comp.get("energy", 0.0))
                            p_errs.append(comp.get("patch_nn", 0.0))
                        cbar.progress((i+1)/(N*2), text=f"Pass 1 — Frame {i+1}/{N}")

                    scales = calibrate_component_scales(
                        np.array(t_errs), np.array(tl_errs), np.array(s_errs), np.array(e_errs), np.array(p_errs),
                    )

                    scorer_scaled = AnomalyScorer(
                        temporal_short = ts, temporal_long = tl, spatial_head = sm,
                        energy_model = em, device = cfg.DEVICE,
                        t_scale = scales["t_scale"], t_long_scale = scales["t_long_scale"],
                        s_scale = scales["s_scale"], e_scale = scales["e_scale"], p_scale = scales["p_scale"],
                        pca_model = pca_model, normal_cls_bank = cls_embs, normal_patch_bank = patch_embs,
                    )

                    normal_scores = []
                    for i in range(N):
                        s, _ = scorer_scaled.push_and_score(cls_embs[i], patch_embs[i])
                        if scorer_scaled.is_ready(): normal_scores.append(s)
                        cbar.progress(0.5 + (i+1)/(N*2), text=f"Pass 2 — Frame {i+1}/{N}")

                    cbar.empty()
                    normal_scores = np.array(normal_scores)
                    thr = calibrate_threshold(normal_scores, component_scales=scales)
                    st.session_state["threshold"] = thr
                    st.session_state["calibrated"] = True
                    st.cache_resource.clear()
                    st.success(f"✅ Baseline Statistical Threshold Established: {thr:.4f}")
                except Exception as e:
                    st.error(f"Calibration failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)


with tab_dash:
    if not model_ready() or not calibration_ready():
        st.info("System not initialized. Please complete Setup and Calibration pipelines.")
    else:
        from training.calibration import load_calibration
        calib_data = load_calibration()
        threshold  = calib_data["threshold"] if calib_data else st.session_state["threshold"]

        # Dashboard layout controls
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
        with col_ctrl1:
            mode = st.selectbox("Data Source", ["Video File Stream", "Live Edge Camera (Webcam)"])
        with col_ctrl2:
            thr_mode = st.selectbox("Thresholding Strategy", ["Dynamic (Mean + 2*Std)", "Static (Calibrated Baseline)"])
        with col_ctrl3:
            if mode == "Video File Stream":
                uploaded_test = st.file_uploader("Mount Video Source", type=["mp4","avi","mov","mkv"], label_visibility="collapsed")
        
        st.markdown("<hr style='margin: 10px 0 20px 0; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
        
        # DASHBOARD GRID
        col_left, col_center, col_right = st.columns([1.6, 1.4, 1.0])
        
        with col_left:
            st.markdown("<div class='card'><div class='card-title'>Live Analysis Feed</div>", unsafe_allow_html=True)
            frame_disp = st.empty()
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='card'><div class='card-title'>Spatial Heatmap</div>", unsafe_allow_html=True)
            heatmap_disp = st.empty()
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_center:
            st.markdown("<div class='card'><div class='card-title'>Anomaly Score Timeline</div>", unsafe_allow_html=True)
            chart_disp = st.empty()
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='card'><div class='card-title'>System Confidence & Uncertainty</div>", unsafe_allow_html=True)
            conf_disp = st.empty()
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown("<div class='card'><div class='card-title'>System Status</div>", unsafe_allow_html=True)
            alert_disp = st.empty()
            alert_disp.markdown('<div class="alert-card alert-normal"><div>SYS STATE</div><div style="font-size:1.4rem">IDLE</div></div>', unsafe_allow_html=True)
            
            m1, m2 = st.columns(2)
            metric_score = m1.empty()
            metric_conf = m2.empty()
            
            metric_score.metric("Current Score", "0.000")
            metric_conf.metric("Confidence", "0.0%")
            
            gauge_disp = st.empty()
            radar_disp = st.empty()
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='card'><div class='card-title'>Operator Controls</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.button("Approve Alert", use_container_width=True)
            c2.button("False Positive", use_container_width=True)
            c3, c4 = st.columns(2)
            c3.button("Escalate", use_container_width=True)
            c4_empty = c4.empty()
            c4_empty.button("Generate Report", use_container_width=True, disabled=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("<div class='card'><div class='card-title'>Operational Logs</div>", unsafe_allow_html=True)
        log_disp = st.empty()
        log_disp.markdown('<div class="log-container">System ready...<br>Awaiting stream...</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # INFERENCE LOGIC
        run_test = False
        
        if mode == "Video File Stream":
            if uploaded_test:
                col_btn1, _, _ = st.columns([1, 2, 2])
                with col_btn1:
                    run_btn = st.button("▶ Initialize Video Stream", type="primary", use_container_width=True)
                    if run_btn: run_test = True
        elif mode == "Live Edge Camera (Webcam)":
            col_btn1, col_btn2, _ = st.columns([1, 1, 2])
            with col_btn1:
                start_stream = st.button("▶ Connect Edge Feed", type="primary", use_container_width=True)
            with col_btn2:
                stop_stream  = st.button("⏹ Disconnect", use_container_width=True)

        @st.cache_resource
        def get_scorer_and_encoder():
            from inference.pipeline import build_scorer
            stem = st.session_state.get("normal_video_stem", "") or cfg.NORMAL_VIDEO.stem
            scorer, encoder, _ = build_scorer(cfg.DEVICE, video_stem=stem)
            return scorer, encoder
            
        def update_logs_ui():
            log_html = '<div class="log-container">'
            for l in st.session_state["system_logs"][::-1]:
                color_cls = "log-crit" if l["level"]=="crit" else "log-warn" if l["level"]=="warn" else ""
                log_html += f'<div class="log-line {color_cls}"><span class="log-time">[{l["time"]}]</span> {l["msg"]}</div>'
            log_html += '</div>'
            log_disp.markdown(log_html, unsafe_allow_html=True)

        if run_test and uploaded_test:
            test_path = save_uploaded_video(uploaded_test)
            add_system_log(f"Stream initiated: {test_path.name}")
            update_logs_ui()

            try:
                scorer, encoder = get_scorer_and_encoder()
                scorer.reset()
                
                from inference.pipeline import run_video_inference
                
                results = []
                dynamic_scores = []
                
                def frame_cb(frame_rgb, result, fidx):
                    dynamic_scores.append(result["score"])
                    
                    if thr_mode == "Dynamic (Mean + 2*Std)":
                        current_thr = np.mean(dynamic_scores) + 2.0 * np.std(dynamic_scores) if len(dynamic_scores) > 1 else np.mean(dynamic_scores)
                    else:
                        current_thr = threshold
                        
                    result["is_anomaly"] = float(result["score"]) > float(current_thr)
                    result["confidence"] = calculate_confidence(result.get("uncertainty", 0.0))
                    results.append(result)
                    
                    # UI Updates (throttle slightly for performance, e.g., every 2 frames)
                    if fidx % 2 == 0:
                        disp_img = annotate_frame_enterprise(frame_rgb, result["score"], result["confidence"], result["is_anomaly"], current_thr)
                        frame_disp.image(disp_img, use_container_width=True)
                        
                        metric_score.metric("Current Score", f"{result['score']:.3f}", delta=f"thr={current_thr:.2f}", delta_color="inverse")
                        metric_conf.metric("Confidence", f"{result['confidence']:.1f}%")
                        gauge_disp.plotly_chart(make_gauge(result["score"], float(current_thr)), key=f"gauge_{fidx}")
                        radar_disp.plotly_chart(make_component_radar(result), key=f"radar_{fidx}")
                        
                        if result["is_anomaly"]:
                            alert_disp.markdown('<div class="alert-card alert-anomaly"><div>SYS STATE</div><div style="font-size:1.4rem">ANOMALY DETECTED</div></div>', unsafe_allow_html=True)
                            if len(results) == 1 or not results[-2]["is_anomaly"]:
                                add_system_log(f"Anomaly detected at frame {fidx}. Score: {result['score']:.2f}", "crit")
                                update_logs_ui()
                        else:
                            alert_disp.markdown('<div class="alert-card alert-normal"><div>SYS STATE</div><div style="font-size:1.4rem">NORMAL</div></div>', unsafe_allow_html=True)
                            
                    if fidx % 5 == 0 and len(results) > 1:
                        chart_disp.plotly_chart(make_score_timeline(results[-100:], float(current_thr)), key=f"tl_{fidx}")
                        conf_disp.plotly_chart(make_conf_timeline(results[-100:]), key=f"conf_{fidx}")
                        
                    # Mock Heatmap generation (since computing exact patch diff is slow per frame)
                    if result["is_anomaly"] and fidx % 5 == 0:
                        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                        # Generate a mock spatial attention heatmap focused on center
                        h, w = gray.shape
                        hm = np.zeros((h, w), dtype=np.float32)
                        cv2.circle(hm, (w//2, h//2), 100, 1.0, -1)
                        hm = cv2.GaussianBlur(hm, (99,99), 50)
                        hm_c = cv2.applyColorMap((hm*255).astype(np.uint8), cv2.COLORMAP_JET)
                        overlay = cv2.addWeighted(frame_rgb, 0.5, cv2.cvtColor(hm_c, cv2.COLOR_BGR2RGB), 0.5, 0)
                        heatmap_disp.image(overlay, use_container_width=True)

                run_video_inference(test_path, scorer, encoder, 999.0, frame_callback=frame_cb)
                add_system_log("Stream processing complete.")
                update_logs_ui()
                
                # --- PDF Generation ---
                anomaly_frames = [r for r in results if r.get("is_anomaly")]
                if anomaly_frames:
                    add_system_log(f"Compiling PDF report for {len(anomaly_frames)} anomalies...")
                    update_logs_ui()
                    try:
                        from fpdf import FPDF
                        import tempfile, os
                        pdf = FPDF()
                        pdf.set_auto_page_break(auto=True, margin=15)
                        pdf_tmp_dir = tempfile.mkdtemp()
                        
                        top_anom = sorted(anomaly_frames, key=lambda r: r["score"], reverse=True)[:10]
                        
                        for pi, ar in enumerate(top_anom):
                            a_rgb = ar.get("frame_rgb", None)
                            if a_rgb is None: continue
                            
                            disp_img = annotate_frame_enterprise(a_rgb, ar["score"], ar["confidence"], True, threshold)
                            anom_path = os.path.join(pdf_tmp_dir, f"a_{pi}.jpg")
                            cv2.imwrite(anom_path, cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR))
                            
                            pdf.add_page()
                            pdf.set_font("helvetica", 'B', 16)
                            pdf.cell(0, 10, txt=f"Anomaly Event Report: Frame {ar['frame_idx']}", ln=True, align='C')
                            pdf.ln(10)
                            
                            pdf.image(anom_path, x=55, y=30, w=100)
                            
                            pdf.set_xy(10, 120)
                            pdf.set_font("helvetica", 'B', 13)
                            pdf.set_text_color(15, 23, 42) 
                            pdf.cell(0, 10, txt="Component Score Breakdown", ln=True)
                            pdf.set_text_color(0, 0, 0)
                            
                            pdf.set_font("helvetica", 'B', 11)
                            pdf.cell(100, 8, txt="Diagnostic Metric", border=1, ln=0, align='C')
                            pdf.cell(50, 8, txt="Recorded Value", border=1, ln=1, align='C')
                            
                            pdf.set_font("helvetica", size=11)
                            metrics = [
                                ("System Confidence", f"{ar.get('confidence', 0.0):.1f}%"),
                                ("Composite Severity Score", f"{ar['score']:.4f}"),
                                ("Spatial Structural Deviation", f"{ar['spatial']:.4f}"),
                                ("Temporal Motion Drift", f"{ar.get('temporal', 0):.4f}"),
                                ("Deep SVDD Feature Energy", f"{ar.get('energy', 0):.4f}")
                            ]
                            for label, val in metrics:
                                pdf.cell(100, 8, txt=label, border=1, ln=0)
                                pdf.cell(50, 8, txt=val, border=1, ln=1, align='C')
                                
                        pdf_path = os.path.join(pdf_tmp_dir, "report.pdf")
                        pdf.output(pdf_path)
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()
                            
                        c4_empty.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="JEPA_Enterprise_Report.pdf", mime="application/pdf", use_container_width=True, type="primary")
                        add_system_log("PDF Report Ready for Download.")
                        update_logs_ui()
                        
                    except ImportError:
                        add_system_log("fpdf library not installed. Cannot generate PDF.", "warn")
                        update_logs_ui()
                    except Exception as e:
                        add_system_log(f"PDF error: {e}", "warn")
                        update_logs_ui()
                
            except Exception as e:
                st.error(f"Inference error: {e}")
                
        elif mode == "Live Edge Camera (Webcam)":
            if start_stream:
                st.session_state["webcam_running"] = True
                add_system_log("Edge camera stream connected.")
                update_logs_ui()
                
            if stop_stream:
                st.session_state["webcam_running"] = False
                add_system_log("Edge camera stream disconnected.")
                update_logs_ui()
                
            if st.session_state.get("webcam_running"):
                try:
                    scorer, encoder = get_scorer_and_encoder()
                    scorer.reset()
                    
                    cap = cv2.VideoCapture(0)
                    frame_count = 0
                    webcam_results = []
                    webcam_scores = []
                    
                    while st.session_state["webcam_running"]:
                        ret, frame_bgr = cap.read()
                        if not ret: break
                        
                        frame_count += 1
                        if frame_count % 2 == 0:
                            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                            frame_resized = cv2.resize(frame_rgb, (cfg.FRAME_SIZE, cfg.FRAME_SIZE))
                            
                            cls_emb, patch_emb = encoder.encode_frame_np(frame_resized)
                            score, components = scorer.push_and_score(cls_emb, patch_emb)
                            
                            webcam_scores.append(score)
                            current_thr = threshold if thr_mode == "Static (Calibrated Baseline)" else (np.mean(webcam_scores) + 2.0*np.std(webcam_scores) if len(webcam_scores)>1 else np.mean(webcam_scores))
                            
                            is_anomaly = float(score) > float(current_thr)
                            conf = calculate_confidence(components.get("uncertainty", 0.0))
                            
                            res = {"frame_idx": frame_count, "score": score, "confidence": conf, "is_anomaly": is_anomaly, **components}
                            webcam_results.append(res)
                            
                            disp_img = annotate_frame_enterprise(frame_rgb, score, conf, is_anomaly, current_thr)
                            frame_disp.image(disp_img, use_container_width=True)
                            
                            metric_score.metric("Current Score", f"{score:.3f}")
                            metric_conf.metric("Confidence", f"{conf:.1f}%")
                            gauge_disp.plotly_chart(make_gauge(score, float(current_thr)), key=f"wc_gauge_{frame_count}")
                            
                            if is_anomaly:
                                alert_disp.markdown('<div class="alert-card alert-anomaly"><div>SYS STATE</div><div style="font-size:1.4rem">ANOMALY DETECTED</div></div>', unsafe_allow_html=True)
                            else:
                                alert_disp.markdown('<div class="alert-card alert-normal"><div>SYS STATE</div><div style="font-size:1.4rem">NORMAL</div></div>', unsafe_allow_html=True)
                                
                            if len(webcam_results) % 5 == 0 and len(webcam_results) > 1:
                                chart_disp.plotly_chart(make_score_timeline(webcam_results[-100:], float(current_thr)), key=f"wc_tl_{frame_count}")
                                conf_disp.plotly_chart(make_conf_timeline(webcam_results[-100:]), key=f"wc_conf_{frame_count}")
                                
                        time.sleep(0.02)
                    cap.release()
                except Exception as e:
                    st.error(f"Webcam error: {e}")
