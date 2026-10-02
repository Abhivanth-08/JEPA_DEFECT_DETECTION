# 2026 JEPA Exhibit Defect Detection

An advanced, generalized anomaly and defect detection system powered by **Joint Embedding Predictive Architectures (JEPA)**. Designed for modern industrial or exhibition environments, this system learns what "normal, defect-free" operations look like purely from unlabeled video sequences. It robustly detects anomalous objects, physical defects, or unusual events in real time.

---

### 🎥 Prototype Demo
**[Watch the live prototype and video demonstration here](https://www.youtube.com/watch?v=QXVEDSoT5Rk)**

---

## 🏗 Key Architectural Features
This pipeline integrates 5 major algorithmic upgrades to deliver state-of-the-art anomaly representations:
1. **Partial ViT Encoder Fine-Tuning**: Adapts a pre-trained Vision Transformer (ViT-B/16) on the target domain, extracting deep spatial embeddings while retaining generalized object knowledge.
2. **Cross-Attention Spatial JEPA**: Compares local spatial patch relationships to establish robust object-level semantic matching, flagging structural or spatial anomalies.
3. **Multi-Scale Temporal Transformers**: Evaluates time sequences using a short window (K=8) for immediate predictive modeling and a long window (K=32) for detecting slow operational drift.
4. **Deep SVDD Energy Model**: Implements Support Vector Data Description (SVDD) to map normal embeddings into a condensed energy hypersphere, making it mathematically strict against outliers.
5. **Uncertainty-Aware Scoring (MC Dropout)**: Runs Monte Carlo dropout sampling to predict model uncertainty. High-variance (noisy) predictions gracefully decay in anomaly strength, preventing false positives tracking anomalies.

---

## 🛠 System Workflows

The platform operates through three seamless stages available in the web dashboard:

1. **🎓 Train on Normal Video**
   - Simply upload a recording of your flawless environment.
   - The backend runs a completely unsupervised, parameter-efficient fine-tuning phase—no manual labels or bounded boxes needed. 
2. **🎯 Dynamic Calibration**
   - Calibrates thresholds by scoring your normal dataset.
   - Normalization scalars and cutoffs (typically taking the 97th percentile of normal deviation) are generated to prevent over-sensitive alerting.
3. **🔍 Detect Anomalies (Test Video / Live Stream)**
   - Upload suspect video sequences or activate your webcam for live tracking.
   - Calculates real-time composite defect scores. 
   - Dynamically highlights regions of interest using patch-difference heatmaps.
   - Auto-generates detailed PDF anomaly reports summarizing spatial vs. temporal failures.

---

## 📂 Project Structure
*All comments have been stripped from the actual Python files to maintain a highly lean footprint for final evaluation.*

- **`app4.py`** — The master Streamlit dashboard orchestrating the end-to-end system.
- **`config.py`** — Global variables, architectural dimension parameters, paths, and hyperparameters.
- **`models/`** — Model implementations containing the ViT Encoder mapping, Cross-Attention Spatial JEPA Head, and Multi-Scale Temporal Transformers.
- **`training/`** — Core unsupervised JEPA learning loops, SVDD hypersphere training routines, multi-loss configurations, and scale calibration schemas.
- **`anomaly/`** — Inference scoring modules which aggregate spatial, temporal, energy, and uncertainty scores into a single composite anomaly metric.
- **`inference/`** — Render and inference pipeline orchestrating cv2 video pipelines directly with the AI models.
- **`data/`** — Data loading strategies and batch samplers.

---

## 🚀 Getting Started

Ensure you have Python 3.9+ installed and a CUDA-capable runtime if you seek to utilize GPU acceleration.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
streamlit run app4.py
```
*Navigate to the local web interface provided by Streamlit. From there, initiate Stage 1 by uploading your baseline video.*
