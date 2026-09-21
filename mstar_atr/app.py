"""
MSTAR SAR Automatic Target Recognition (ATR) - Tactical Radar Console.
Streamlit Interactive Application for SAR Image Analysis, Model Inference, and Grad-CAM Explainability.
"""

import os
import sys
from typing import Optional, Tuple
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F

from mstar_atr.constants import (
    CLASSES,
    DEFAULT_CROP_SIZE,
    TARGET_METADATA,
)
from mstar_atr.data.dataset import SARTransform, create_synthetic_mstar_dataset
from mstar_atr.filters.speckle import (
    amplitude_to_db,
    frost_filter,
    lee_filter,
    median_filter,
    normalize_image,
)
from mstar_atr.interpretation.gradcam import GradCAM, overlay_gradcam_on_sar
from mstar_atr.models.aconvnet import AConvNet
from mstar_atr.models.legacy_cnn import LegacyMSTARCNN
from mstar_atr.models.resnet import build_resnet18_sar


# ---------------------------------------------------------
# Page Configuration & Tactical Radar Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="MSTAR SAR Tactical ATR Console",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Tactical Radar CSS styling
st.markdown(
    """
    <style>
    /* Dark Tactical Theme */
    .stApp {
        background: linear-gradient(135deg, #070b14 0%, #0d1527 50%, #070b14 100%);
        color: #e2e8f0;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Header Card */
    .radar-header {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(0, 229, 255, 0.25);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .radar-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00ff9d, #00e5ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: 1px;
    }
    
    .radar-subtitle {
        color: #94a3b8;
        font-size: 1.0rem;
        margin-top: 0.3rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: rgba(0, 255, 157, 0.15);
        border: 1px solid #00ff9d;
        color: #00ff9d;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    /* Image Cards */
    .panel-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    
    .panel-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    /* Tactical Specs Card */
    .tactical-card {
        background: rgba(15, 23, 42, 0.85);
        border-left: 4px solid #00e5ff;
        border-radius: 8px;
        padding: 1.25rem;
        margin-top: 1rem;
    }
    
    .tactical-name {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.25rem;
    }
    
    .tactical-role {
        font-size: 0.9rem;
        color: #00ff9d;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    
    .spec-row {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1);
        font-size: 0.85rem;
    }
    
    .spec-label {
        color: #94a3b8;
    }
    
    .spec-value {
        color: #f1f5f9;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Caching & Model Initializer
# ---------------------------------------------------------
@st.cache_resource
def get_sample_assets_dir() -> str:
    """Ensures sample chips exist and returns sample directory path."""
    samples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "samples")
    if not os.path.isdir(samples_dir) or len(os.listdir(samples_dir)) == 0:
        os.makedirs(samples_dir, exist_ok=True)
        # Create a sample chip for each class
        create_synthetic_mstar_dataset(
            output_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "samples_temp"),
            samples_per_class=2,
        )
        temp_train = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "samples_temp", "train")
        for cls_name in CLASSES:
            cls_src = os.path.join(temp_train, cls_name)
            if os.path.isdir(cls_src):
                files = os.listdir(cls_src)
                if files:
                    src_f = os.path.join(cls_src, files[0])
                    dst_f = os.path.join(samples_dir, f"{cls_name}_sample.jpeg")
                    Image.open(src_f).save(dst_f)
    return samples_dir


@st.cache_resource
def load_cached_model(arch_name: str, checkpoint_path: Optional[str] = None) -> torch.nn.Module:
    """Instantiates and optionally loads model weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = arch_name.lower().replace("-", "").replace("_", "")

    if arch in ["aconvnet", "aconv"]:
        model = AConvNet(num_classes=len(CLASSES), in_channels=1)
    elif arch in ["resnet", "resnet18"]:
        model = build_resnet18_sar(num_classes=len(CLASSES), in_channels=1)
    else:
        model = LegacyMSTARCNN(num_classes=len(CLASSES), in_channels=1)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------
# Header & Navigation
# ---------------------------------------------------------
samples_dir = get_sample_assets_dir()

st.markdown(
    """
    <div class="radar-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 class="radar-title">📡 MSTAR SAR RADAR ATR CONSOLE</h1>
                <div class="radar-subtitle">
                    Synthetic Aperture Radar Automatic Target Recognition • Physics-Aware Deep Learning & Explainability
                </div>
            </div>
            <div>
                <span class="status-badge">● SYSTEM ACTIVE</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigation Tabs
tab_radar, tab_benchmark, tab_physics = st.tabs(
    ["🎯 Live Target Analysis", "📊 Benchmark Comparison", "🔬 SAR Physics & Theory"]
)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ Radar Sensor & Model Setup")

model_choice = st.sidebar.selectbox(
    "Architecture",
    ["A-ConvNet (SAR SOTA)", "ResNet-18 SAR", "Legacy 3-Layer CNN (2021)"],
    index=0,
)

checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")] if os.path.isdir(checkpoint_dir) else []

selected_ckpt = None
if checkpoint_files:
    ckpt_option = st.sidebar.selectbox("Checkpoint", ["None (Untrained / Scratch)"] + checkpoint_files)
    if ckpt_option != "None (Untrained / Scratch)":
        selected_ckpt = os.path.join(checkpoint_dir, ckpt_option)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛰️ Input Image Selection")
input_mode = st.sidebar.radio("Input Source", ["MSTAR Target Presets", "Upload SAR Image"])

raw_image_array = None
selected_class_label = "2S1"

if input_mode == "MSTAR Target Presets":
    sample_files = sorted(os.listdir(samples_dir)) if os.path.isdir(samples_dir) else []
    sample_classes = [s.split("_")[0] for s in sample_files if "_" in s]
    target_cls = st.sidebar.selectbox("Target Class Preset", CLASSES, index=1)  # BMP2 by default
    selected_class_label = target_cls

    matching_files = [f for f in sample_files if f.startswith(f"{target_cls}_")]
    if matching_files:
        sample_path = os.path.join(samples_dir, matching_files[0])
        img = Image.open(sample_path).convert("L")
        raw_image_array = np.array(img, dtype=np.float32)
    else:
        # Fallback generate chip
        temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "temp_single")
        create_synthetic_mstar_dataset(temp_dir, samples_per_class=1)
        fallback_path = os.path.join(temp_dir, "train", target_cls, f"{target_cls}_train_000.jpeg")
        if os.path.isfile(fallback_path):
            img = Image.open(fallback_path).convert("L")
            raw_image_array = np.array(img, dtype=np.float32)
else:
    uploaded = st.sidebar.file_uploader("Upload SAR Chip (JPEG, PNG, TIFF, NPY)", type=["png", "jpg", "jpeg", "tif", "npy"])
    if uploaded is not None:
        if uploaded.name.endswith(".npy"):
            raw_image_array = np.load(uploaded).astype(np.float32)
        else:
            img = Image.open(uploaded).convert("L")
            raw_image_array = np.array(img, dtype=np.float32)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ SAR Signal Processing")
use_db = st.sidebar.toggle("Dynamic Range dB Scaling", value=True)
speckle_filter_choice = st.sidebar.selectbox(
    "Speckle Filter",
    ["None", "Lee Filter (Adaptive)", "Frost Filter (Exponential)", "Median Filter"],
    index=1,
)

filter_window = st.sidebar.slider("Filter Window Size", min_value=3, max_value=9, value=5, step=2)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Grad-CAM Explainability")
cam_alpha = st.sidebar.slider("Heatmap Blend Alpha", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
colormap_name = st.sidebar.selectbox("Colormap", ["inferno", "turbo", "plasma", "jet", "viridis"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 References & Credits")
st.sidebar.markdown(
    """
    1. [hunterlew/mstar_with_machine_learning](https://github.com/hunterlew/mstar_with_machine_learning)
    2. [shuibao/CNN_MSTAR](https://github.com/shuibao/CNN_MSTAR)
    3. [DARPA/AFRL MSTAR Benchmark](https://www.sdms.afrl.af.mil/)
    """
)


# ---------------------------------------------------------
# Tab 1: Live Target Analysis
# ---------------------------------------------------------
with tab_radar:
    if raw_image_array is None:
        st.info("Please select or upload a SAR image to analyze.")
    else:
        # Process image
        processed_array = raw_image_array.copy()

        # Apply speckle filter
        if speckle_filter_choice == "Lee Filter (Adaptive)":
            processed_array = lee_filter(processed_array, window_size=filter_window)
        elif speckle_filter_choice == "Frost Filter (Exponential)":
            processed_array = frost_filter(processed_array, window_size=filter_window)
        elif speckle_filter_choice == "Median Filter":
            processed_array = median_filter(processed_array, window_size=filter_window)

        # Apply dB scaling or normalization
        if use_db:
            display_processed = amplitude_to_db(processed_array)
        else:
            display_processed = normalize_image(processed_array)

        # Prepare PyTorch Tensor for inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_cached_model(model_choice, selected_ckpt)

        transform = SARTransform(output_size=DEFAULT_CROP_SIZE, is_training=False, use_db_scale=use_db)
        input_tensor = transform(processed_array).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        # Compute Grad-CAM
        try:
            cam_engine = GradCAM(model)
            heatmap, _, _ = cam_engine.generate_heatmap(input_tensor, target_class=pred_idx)
            gradcam_overlay = overlay_gradcam_on_sar(
                raw_image_array,
                heatmap,
                alpha=cam_alpha,
                colormap_name=colormap_name,
            )
            cam_engine.remove_hooks()
        except Exception as e:
            gradcam_overlay = np.stack([normalize_image(raw_image_array, to_uint8=True)] * 3, axis=-1)

        # Display Top 3 Columns: Raw SAR, Filtered/Enhanced, Grad-CAM Overlay
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="panel-card"><div class="panel-label">Raw SAR Amplitude Chip</div>', unsafe_allow_html=True)
            st.image(normalize_image(raw_image_array, to_uint8=True), use_container_width=True, caption=f"Raw Sensor Return ({raw_image_array.shape[0]}x{raw_image_array.shape[1]})")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="panel-card"><div class="panel-label">Enhanced ({speckle_filter_choice} + dB)</div>', unsafe_allow_html=True)
            st.image((display_processed * 255).astype(np.uint8), use_container_width=True, caption="Dynamic Range Compressed & Denoised")
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="panel-card"><div class="panel-label">Grad-CAM Radar Attention Map</div>', unsafe_allow_html=True)
            st.image(gradcam_overlay, use_container_width=True, caption=f"Focus on {pred_class} Scatterers & Dihedrals")
            st.markdown("</div>", unsafe_allow_html=True)

        # Classification Results & Tactical Metadata
        st.markdown("---")
        res_col1, res_col2 = st.columns([1.2, 1.0])

        with res_col1:
            st.markdown("### 🎯 Automatic Target Classification")
            meta = TARGET_METADATA.get(pred_class, {})

            st.markdown(
                f"""
                <div class="tactical-card">
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <div class="tactical-name">{meta.get('full_name', pred_class)}</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #00ff9d;">{confidence * 100:.1f}% Match</div>
                    </div>
                    <div class="tactical-role">{meta.get('category', 'Military Target')} • {meta.get('origin', 'N/A')}</div>
                    <div class="spec-row">
                        <span class="spec-label">Hull Dimensions</span>
                        <span class="spec-value">{meta.get('length_width', 'N/A')}</span>
                    </div>
                    <div class="spec-row">
                        <span class="spec-label">Combat Weight</span>
                        <span class="spec-value">{meta.get('weight', 'N/A')}</span>
                    </div>
                    <div class="spec-row">
                        <span class="spec-label">Radar Scatterer Signature</span>
                        <span class="spec-value" style="font-size: 0.8rem; text-align: right; max-width: 65%;">{meta.get('radar_signature', 'N/A')}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col2:
            st.markdown("### 📊 Probability Distribution")
            chart_data = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
            st.bar_chart(chart_data)


# ---------------------------------------------------------
# Tab 2: Benchmark Comparison
# ---------------------------------------------------------
with tab_benchmark:
    st.markdown("### 🏆 MSTAR 10-Class Benchmark Results (Standard Operating Conditions)")
    st.markdown(
        """
        The standard benchmark protocol trains on chips acquired at **17° depression angle** (~2,049 chips)
        and tests on chips acquired at **15° depression angle** (~1,838 chips).
        """
    )

    benchmark_data = [
        {"Model Architecture": "A-ConvNet (All-Convolutional)", "Framework": "PyTorch", "Parameters": "0.38 M", "Accuracy (SOC)": "99.13%", "Speckle Robustness": "Very High", "Inference Latency": "2.1 ms"},
        {"Model Architecture": "ResNet-18 SAR", "Framework": "PyTorch", "Parameters": "11.17 M", "Accuracy (SOC)": "98.45%", "Speckle Robustness": "High", "Inference Latency": "3.8 ms"},
        {"Model Architecture": "Support Vector Machine (RBF + PCA-80)", "Framework": "Scikit-Learn", "Parameters": "N/A", "Accuracy (SOC)": "97.81%", "Speckle Robustness": "Moderate", "Inference Latency": "0.8 ms"},
        {"Model Architecture": "Random Forest (1000 trees)", "Framework": "Scikit-Learn", "Parameters": "N/A", "Accuracy (SOC)": "96.49%", "Speckle Robustness": "Moderate", "Inference Latency": "4.2 ms"},
        {"Model Architecture": "Gradient Boosted Trees (GBDT)", "Framework": "Scikit-Learn", "Parameters": "N/A", "Accuracy (SOC)": "95.17%", "Speckle Robustness": "Moderate", "Inference Latency": "3.5 ms"},
        {"Model Architecture": "Original 3-Layer CNN (2021 Baseline)", "Framework": "Keras / PyTorch", "Parameters": "0.45 M", "Accuracy (SOC)": "91.20%", "Speckle Robustness": "Low (Overfits noise)", "Inference Latency": "1.9 ms"},
        {"Model Architecture": "Decision Tree (Entropy)", "Framework": "Scikit-Learn", "Parameters": "N/A", "Accuracy (SOC)": "70.68%", "Speckle Robustness": "Poor", "Inference Latency": "0.3 ms"},
    ]

    st.dataframe(benchmark_data, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 💡 Key Takeaway: Why A-ConvNet Outperforms Traditional Deep CNNs")
    st.markdown(
        """
        In natural RGB computer vision, deep architectures with huge fully connected layers excel because natural scenes have rich texture.
        In **SAR imagery**, fully connected layers end up memorizing high-frequency speckle noise patterns rather than target physical features.
        **A-ConvNet** completely removes fully connected layers, replacing them with convolutional layers with stride and global pooling, forcing the network to learn translation-invariant radar scattering centers and dihedral reflections.
        """
    )


# ---------------------------------------------------------
# Tab 3: Radar Physics & Theory
# ---------------------------------------------------------
with tab_physics:
    st.markdown("### 🔬 SAR Radar Physics Primer for Machine Learning Engineers")

    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st.markdown(
            """
            #### 1. Coherent Radar Speckle
            Synthetic Aperture Radar transmits coherent microwave pulses (e.g. X-band 9.6 GHz in MSTAR).
            Because the resolution cell (~0.3m x 0.3m) contains multiple sub-wavelength scatterers, the returned echoes interfere constructively and destructively.
            This creates **multiplicative speckle noise**, which follows a Rayleigh or Gamma distribution rather than standard additive Gaussian noise.
            
            #### 2. Specular & Dihedral Scattering
            Military ground targets are composed of metallic plates.
            - **Dihedral reflectors** (e.g., ground plate meeting vertical vehicle side, or bulldozer blade meeting hull) create intense double-bounce returns, appearing as bright white spots.
            - **Flat armor surfaces** reflect radar beams away from the receiver unless perpendicular, appearing dark.
            """
        )

    with p_col2:
        st.markdown(
            """
            #### 3. Radar Shadows
            Because the radar illuminates the target from an oblique depression angle (15° - 17°), the target body physically blocks microwaves from reaching the terrain behind it.
            This creates a **radar shadow cavity** directly behind the vehicle with zero backscatter.
            In SAR ATR, the geometry of the shadow is just as informative about the vehicle's height and turret silhouette as the bright scatterers!
            
            #### 4. SOC vs. EOC Operating Conditions
            - **Standard Operating Conditions (SOC)**: Training and test sets have identical target configurations and only minor depression angle variation (17° vs 15°).
            - **Extended Operating Conditions (EOC)**: Models are evaluated against large depression angle changes (e.g. 30°/45°) or target configuration changes (e.g. tanks with/without auxiliary fuel drums and reactive armor).
            """
        )

# ---------------------------------------------------------
# Persistent Footer with Credits & References
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 1.25rem 0; line-height: 1.6;">
        <span style="font-weight: 700; color: #f1f5f9;">MSTAR SAR Tactical Automatic Target Recognition Console</span> • Modernized by Rohit Singh (<a href="https://github.com/rohitgit1" target="_blank" style="color: #38bdf8; text-decoration: none;">@rohitgit1</a>)<br>
        <span style="font-weight: 600; color: #cbd5e1;">References & Credits:</span> 
        <a href="https://github.com/hunterlew/mstar_with_machine_learning" target="_blank" style="color: #00ff9d; text-decoration: none; margin: 0 8px;">1. hunterlew/mstar_with_machine_learning</a> • 
        <a href="https://github.com/shuibao/CNN_MSTAR" target="_blank" style="color: #00ff9d; text-decoration: none; margin: 0 8px;">2. shuibao/CNN_MSTAR</a>
    </div>
    """,
    unsafe_allow_html=True,
)

