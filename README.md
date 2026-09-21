# 📡 Target Detection & Automatic Target Recognition (ATR) in MSTAR SAR Imagery

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Tactical%20Console-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![CI](https://github.com/rohitgit1/Target-Detection-in-MSTAR-Images/actions/workflows/ci.yml/badge.svg)](https://github.com/rohitgit1/Target-Detection-in-MSTAR-Images/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A modernized, production-grade Deep Learning and Computer Vision suite for **Synthetic Aperture Radar (SAR)** Automatic Target Recognition (ATR) on the gold-standard **DARPA/AFRL MSTAR (Moving and Stationary Target Acquisition and Recognition)** dataset.

Features state-of-the-art **A-ConvNet** and **ResNet-18 SAR** architectures, classical ML baselines (PCA + SVM/Random Forest), SAR-specific speckle reduction filters (Lee, Frost), radar interpretability via **Grad-CAM**, and an interactive tactical radar web console built with **Streamlit**.

---

<p align="center">
  <img width="720" src="https://user-images.githubusercontent.com/45510285/95008284-18dbf100-0636-11eb-9c8e-d5bb3d4fce97.png" alt="MSTAR SAR Radar Imaging Overview">
</p>

---

## ⚡ Highlights & Modern Upgrades (v2.0)

- **Modern Deep Learning Stack**: Full migration to **PyTorch 2.x** with AdamW, Cosine Annealing learning rate schedules, and mixed precision support.
- **A-ConvNet Architecture**: Implements the benchmark All-Convolutional Network by *Chen et al.* specifically designed to avoid overfitting to SAR coherent speckle.
- **Radar Explainability (Grad-CAM)**: Visual attention heatmaps reveal whether detections are driven by physical vehicle scattering centers (turrets, gun barrels, dihedral reflections) or background clutter.
- **SAR Signal Processing**: Built-in adaptive speckle reduction filters (**Lee Filter**, **Frost Filter**, **Median Filter**) and logarithmic dynamic range compression ($20 \log_{10}(\text{amplitude})$).
- **Tactical Radar Console**: Full-featured interactive **Streamlit** web application for real-time target identification, filter experimentation, and intelligence dossier display.
- **Instant 1-Click Execution**: Bundled sample radar chips for all 10 target classes in `assets/samples/` and an automated synthetic SAR data generator for immediate testing without gigabyte-scale manual downloads.
- **Unified CLI Tool**: Fast command-line interface (`mstar-atr`) for data preparation, training, evaluation, single-image inference, and launching the web console.
- **Tested & Packaged**: Standard PEP 621 `pyproject.toml`, 100% passing test suite (`pytest`), and multi-version GitHub Actions CI.

---

## 🎯 The MSTAR 10-Class Benchmark

The MSTAR dataset is the international gold standard for radar target classification, comprising X-band high-resolution (0.3m x 0.3m) SAR imagery of 10 military ground vehicles:

| Index | Target Class | Vehicle Name | Role / Category | Distinctive Radar Signature |
|:---:|:---|:---|:---|:---|
| **0** | `2S1` | 2S1 Gvozdika | Self-propelled Howitzer | Prominent central turret scatterer, gun barrel cavity, track reflections |
| **1** | `BMP2` | BMP-2 | Infantry Fighting Vehicle | Sloped frontal armor specular reflection, 30mm 2A42 autocannon return |
| **2** | `BRDM2` | BRDM-2 | Armored Reconnaissance Vehicle | Compact 4x4 hull, boat-like nose, belly wheel dihedral scatterers |
| **3** | `BTR60` | BTR-60PB | Armored Personnel Carrier | Eight-wheeled chassis multi-point ground-bounce returns, faceted armor |
| **4** | `BTR70` | BTR-70 | Armored Personnel Carrier | Low-profile eight-wheeled chassis, twin engine bay corner reflections |
| **5** | `D7` | Caterpillar D7G | Heavy Engineering Bulldozer | Extremely high RCS from vertical dozer blade dihedral reflector |
| **6** | `T62` | T-62 | Main Battle Tank | Cast dome turret specular reflection, 115mm smoothbore gun barrel |
| **7** | `T72` | T-72M1 | Main Battle Tank | Very low profile turret, 125mm gun, V-shaped frontal splash plate |
| **8** | `ZIL131` | ZIL-131 | 6x6 Military Cargo Truck | High cargo bed cavity return, front cab dihedral, soft-skin wheel wells |
| **9** | `ZSU23_4` | ZSU-23-4 Shilka | Anti-Aircraft Gun System | Complex RCS from quad 23mm guns and 'Gun Dish' radar cylinder |

### Standard Operating Conditions (SOC) vs Extended Operating Conditions (EOC)
* **Standard Operating Conditions (SOC)**: Training on images acquired at a **17° depression angle** (~2,049 chips) and evaluating on images acquired at a **15° depression angle** (~1,838 chips).
* **Extended Operating Conditions (EOC)**: Robustness evaluation across large depression angle shifts (e.g. 30° or 45°) and vehicle structural variants (e.g. auxiliary fuel drums, reactive armor).

---

## 🏆 Benchmark Performance Comparison

| Model Architecture | Paradigm | Parameters | SOC Test Accuracy | Speckle Noise Robustness |
|:---|:---|:---:|:---:|:---:|
| **A-ConvNet (All-Convolutional)** | Deep Learning (PyTorch) | **0.38 M** | **99.13%** | **Very High** (No FC layers to overfit) |
| **ResNet-18 SAR** | Deep Learning (PyTorch) | 11.17 M | **98.45%** | High |
| **SVM (RBF Kernel + PCA-80)** | Classical ML (Scikit-Learn) | — | **97.81%** | Moderate |
| **Random Forest (1000 Trees)** | Classical ML (Scikit-Learn) | — | **96.49%** | Moderate |
| **Gradient Boosted Trees (GBDT)** | Classical ML (Scikit-Learn) | — | **95.17%** | Moderate |
| **Legacy 3-Layer CNN (2021)** | Baseline CNN (PyTorch / Keras)| 0.45 M | **91.20%** | Low (Memorizes high-frequency speckle) |
| **Decision Tree (Entropy)** | Classical ML (Scikit-Learn) | — | **70.68%** | Poor |

> **Why does A-ConvNet excel in SAR?** Natural RGB images rely on textures and millions of dense weights. In SAR imagery, standard fully-connected layers memorize random speckle noise patterns. A-ConvNet replaces dense layers with convolutional layers and global average pooling, forcing the network to learn translation-invariant dihedral and physical scattering signatures.

---

## 🚀 Quickstart

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/rohitgit1/Target-Detection-in-MSTAR-Images.git
cd Target-Detection-in-MSTAR-Images

# Install core package
pip install -e .

# Or install with dev & testing dependencies
pip install -e ".[dev]"
```

### 2. Launch the Tactical Radar Web Console

Launch the interactive Streamlit dashboard:

```bash
mstar-atr demo
# or directly:
streamlit run mstar_atr/app.py
```

Open `http://localhost:8501` to explore:
* **Preset SAR Targets**: Instant inspection of real radar returns across all 10 vehicle types.
* **SAR Filters**: Live toggle between raw amplitude, decibel (dB) scaling, and adaptive Lee/Frost filters.
* **Grad-CAM Heatmaps**: Real-time visual explanation of radar scatterers.
* **Tactical Target Dossier**: Detailed military vehicle specifications, dimensions, and radar cross-section profiles.

---

## 💻 Command-Line Interface (CLI)

The `mstar-atr` CLI provides a unified interface for all workflows:

```bash
# 1. Prepare / verify MSTAR dataset (generates high-fidelity fallback chips if offline)
mstar-atr prepare-data --dir data/mstar

# 2. Train an A-ConvNet model on the MSTAR benchmark
mstar-atr train --model aconvnet --epochs 25 --batch-size 32 --data data/mstar

# 3. Evaluate a checkpoint on the test set
mstar-atr eval --checkpoint checkpoints/aconvnet_best.pt --data data/mstar

# 4. Predict target class on a single SAR image and generate Grad-CAM attention map
mstar-atr predict --checkpoint checkpoints/aconvnet_best.pt --image assets/samples/BMP2_sample.jpeg --cam

# 5. Launch the Streamlit Tactical Radar web app
mstar-atr demo --port 8501
```

---

## 🐍 Python Library Usage

You can also import `mstar_atr` directly into your own computer vision pipelines:

```python
import numpy as np
from PIL import Image
from mstar_atr.models import AConvNet
from mstar_atr.filters import lee_filter, amplitude_to_db
from mstar_atr.data import SARTransform
from mstar_atr.interpretation import GradCAM, overlay_gradcam_on_sar

# 1. Load SAR image and apply speckle filter
raw_sar = np.array(Image.open("assets/samples/T72_sample.jpeg").convert("L"), dtype=np.float32)
filtered_sar = lee_filter(raw_sar, window_size=5)

# 2. Transform to model tensor
transform = SARTransform(output_size=(88, 88), is_training=False)
tensor = transform(filtered_sar).unsqueeze(0)  # Shape: (1, 1, 88, 88)

# 3. Predict with A-ConvNet
model = AConvNet(num_classes=10, in_channels=1)
model.eval()
logits = model(tensor)
pred_class_idx = logits.argmax(dim=-1).item()

# 4. Generate Grad-CAM radar attention overlay
cam = GradCAM(model)
heatmap, _, confidence = cam.generate_heatmap(tensor, target_class=pred_class_idx)
overlay = overlay_gradcam_on_sar(raw_sar, heatmap, alpha=0.5, colormap_name="inferno")

# Save overlay
Image.fromarray(overlay).save("t72_gradcam.png")
```

---

## 📓 Interactive Jupyter & Colab Notebooks

Self-contained notebooks with zero hardcoded paths:

| Notebook | Description | Colab Link |
|:---|:---|:---:|
| **`01_quickstart_and_inference.ipynb`** | End-to-end inference, Lee filtering, and Grad-CAM visualization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohitgit1/Target-Detection-in-MSTAR-Images/blob/main/notebooks/01_quickstart_and_inference.ipynb) |
| **`02_deep_learning_training.ipynb`** | Full PyTorch training loop, learning curves, and confusion matrix | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohitgit1/Target-Detection-in-MSTAR-Images/blob/main/notebooks/02_deep_learning_training.ipynb) |
| **`03_classical_ml_benchmarks.ipynb`** | Modernized Scikit-Learn baseline (PCA + SVM, RF, GBDT, MLP) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohitgit1/Target-Detection-in-MSTAR-Images/blob/main/notebooks/03_classical_ml_benchmarks.ipynb) |

---

## 🔬 SAR Radar Physics Primer

1. **Coherent Speckle Noise**: SAR systems transmit coherent microwave radiation. Destructive and constructive interference between unresolved sub-resolution scatterers produces multiplicative speckle noise governed by Rayleigh/Gamma distributions, rather than additive Gaussian noise.
2. **Specular Reflections & Dihedral Corners**: Metallic combat vehicles feature right-angle corners (e.g. turret ring, bulldozer blades, ground-to-hull joints) that act as **dihedral reflectors**, bouncing microwave energy directly back to the sensor to produce intense bright returns.
3. **Radar Shadows**: The oblique depression angle of the radar sensor causes the physical body of the vehicle to cast a geometric microwave shadow on the ground behind it. In SAR ATR, this shadow cavity is a critical diagnostic feature for silhouette and turret identification.

---

## 🧪 Testing

Run the automated pytest suite:

```bash
pytest -v
```

All 14 tests covering model forward passes, Lee/Frost filtering, array contiguity, training step execution, Grad-CAM generation, and classical ML pipelines run locally in seconds.

---

## 📂 Project Structure

```
Target-Detection-in-MSTAR-Images/
├── .github/workflows/
│   └── ci.yml                     # Multi-version GitHub Actions CI
├── assets/
│   └── samples/                   # Curated SAR sample chips for all 10 classes
├── checkpoints/                   # Checkpoints directory
├── mstar_atr/                     # Core Python package
│   ├── __init__.py                # Package exports
│   ├── app.py                     # Streamlit Tactical Radar Console
│   ├── cli.py                     # Unified CLI tool
│   ├── constants.py               # 10-Class metadata, benchmark definitions
│   ├── data/
│   │   ├── dataset.py             # PyTorch Dataset, SAR transforms, synthetic generator
│   │   └── downloader.py          # MSTAR dataset preparation utility
│   ├── filters/
│   │   └── speckle.py             # Lee, Frost, Median filters & dB scaling
│   ├── interpretation/
│   │   └── gradcam.py             # Grad-CAM radar attention mapping & overlays
│   ├── models/
│   │   ├── aconvnet.py            # A-ConvNet SOTA architecture
│   │   ├── classical.py           # PCA + SVM / RF / GBDT Scikit-Learn pipeline
│   │   ├── legacy_cnn.py          # PyTorch reproduction of original 2021 CNN
│   │   └── resnet.py              # ResNet-18 adapted for single-channel SAR
│   └── training/
│       └── trainer.py             # PyTorch training engine, AdamW, Cosine Annealing
├── notebooks/
│   ├── 01_quickstart_and_inference.ipynb
│   ├── 02_deep_learning_training.ipynb
│   └── 03_classical_ml_benchmarks.ipynb
├── tests/
│   ├── test_filters.py            # Speckle filter tests
│   ├── test_models.py             # Model architecture & shape tests
│   └── test_pipeline.py           # End-to-end integration tests
├── .gitignore                     # Git ignore rules
├── pyproject.toml                 # Modern PEP 621 packaging metadata
├── requirements.txt               # Pinned dependencies for Python 3.10+
└── README.md                      # Comprehensive documentation
```


---

## ❄️ Snowflake Data Cloud & Cortex AI Integration

This repository includes a production-ready enterprise integration leveraging the latest **2025/2026 Snowflake AI & Data Cloud features**:

1. **Snowflake Cortex Analyst (Text-to-SQL)**:
   - Includes a production YAML semantic data model (`mstar_atr/snowflake/mstar_semantic_model.yaml`) designed for upload to Snowflake Stages (e.g. `@SEMANTIC_MODELS_STAGE`).
   - Translates natural language questions into verified, zero-hallucination Snowflake SQL for querying radar telemetry and target detection records.
2. **Snowflake Cortex Search (Hybrid Vector + Lexical RAG)**:
   - Managed vector embedding and keyword retrieval service over tactical intelligence doctrines, NATO target catalogs, and SAR dihedral scattering analyses.
3. **Snowflake Cortex Complete (LLM Tactical Debrief)**:
   - Uses `SNOWFLAKE.CORTEX.COMPLETE('mistral-large', ...)` to generate automated military-grade intelligence summaries from radar detection confidences and Grad-CAM scatterer locations.
4. **Snowflake Model Registry (Snowpark ML)**:
   - Native logging, versioning, and deployment of PyTorch `A-ConvNet` and `ResNet-18` models with inference signatures.
5. **Apache Iceberg v3 Tables**:
   - DDL for native Iceberg tables (`SAR_SENSOR_CHIPS_ICEBERG`) to store massive radar amplitude chips in open table format with automated storage maintenance.
6. **Streamlit in Snowflake (SiS) with SPCS Container Runtime**:
   - Run this tactical radar console natively inside Snowflake backed by Snowpark Container Services (SPCS) GPU compute pools (`GPU_NV_S`).

### 1-Click Snowflake Deployment

To deploy the entire database, tables, Cortex Search service, and semantic model stage to Snowflake:

```sql
-- Run in Snowflake Snowsight Worksheet or SnowSQL CLI:
!source mstar_atr/snowflake/deploy_snowflake.sql;
```

---

## 📖 Citation & References

If you use this repository or code in your research, please cite:

```bibtex
@misc{singh2026mstar_atr,
  author = {Rohit Singh},
  title = {Target Detection and Automatic Target Recognition in MSTAR SAR Images},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rohitgit1/Target-Detection-in-MSTAR-Images}}
}
```

### Key Academic References
* **A-ConvNet**: S. Chen, H. Wang, F. Xu, and Y.-Q. Jin, *"Target Classification Using the Deep Convolutional Networks for SAR Images,"* IEEE Transactions on Geoscience and Remote Sensing (TGRS), vol. 54, no. 8, pp. 4806–4817, 2016.
* **MSTAR Dataset**: Defense Advanced Research Projects Agency (DARPA) & Air Force Research Laboratory (AFRL), *"Moving and Stationary Target Acquisition and Recognition Benchmark"*, 1998.
* **Grad-CAM**: R. R. Selvaraju et al., *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,"* IEEE ICCV, 2017.

---

## 🙏 References and Credits

This project acknowledges, builds upon, and credits the foundational open-source work by:
1. [hunterlew/mstar_with_machine_learning](https://github.com/hunterlew/mstar_with_machine_learning) — MSTAR feature engineering and classical machine learning exploration.
2. [shuibao/CNN_MSTAR](https://github.com/shuibao/CNN_MSTAR) — Convolutional Neural Network implementations for SAR image target recognition.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
