# Deep Learning–Driven Soil Health Assessment Using Camera-Based Image Processing

> **A smartphone-deployable soil analysis system combining EfficientNetV2 classification with spectral index–based property estimation**

---

## Authors

| Name | ID |
|---|---|
| Anurag Paul | 22MID0080 |
| Nimish Sharma | 22MID0087 |
| Kinjal Ghosh | 22MID0331 |

**Domain:** Artificial Intelligence · Deep Learning · Computer Vision · Agricultural Informatics · Explainable AI

---

## Overview

Traditional soil health assessment relies on laboratory analysis — slow, expensive, and inaccessible to small-scale farmers. This project introduces an end-to-end deep learning pipeline that uses **standard RGB images from smartphones** to classify soil type and estimate key physical properties in real time, with no lab equipment required.

The system combines:
- **EfficientNetV2** for high-accuracy soil type classification
- **Spectral index–based visual feature engine** for property estimation (moisture, salinity, OM, pH)
- **A Gradio web interface** for instant interactive analysis

---

## System Architecture

![Block Diagram](Block%20Diagram.svg)

The pipeline operates in three stages:

```
RGB Image Input
      │
      ▼
┌─────────────────────┐
│ 🔬 Auto-Calibration │  Lighting Detection → CLAHE Exposure → Gray World WB
│ (Self-Calibrating)  │  → Calibrated Image + Calibration Report
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│   Preprocessing     │  Resize → Normalize → EfficientNetV2 preprocess_input
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ CNN Perception      │  EfficientNetV2 → Soil Type + Confidence
│ (EfficientNetV2)    │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Visual Feature      │  Brightness Index (Mathieu 1998)
│ Engine              │  Redness Index (Escadafal 1989)
│                     │  Coloration Index (Escadafal 1993)
│                     │  HSV Saturation
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Property Estimation │  Moisture  (Gomez et al. 2008)
│ (Research Formulas) │  Salinity  (Metternicht & Zinck 2003)
│                     │  OM Index  (Konen 2003, Viscarra Rossel 2012)
│                     │  pH        (Viscarra Rossel 2006, Barron & Torrent 1986)
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Soil Health         │  Multi-criterion scoring vs. soil-type KB
│ Assessment          │  → Excellent / Good / Fair / Poor
└─────────────────────┘
      │
      ▼
   Final Report + Individual Image Results + Calibration Report
```

---

## Features

- 🔬 **Self-Calibrating AI** — Automatic lighting detection & image normalization (CLAHE + Gray World) before classification
- 🌱 **Soil Type Classification** — 7 classes: Alluvial, Black, Laterite, Red, Yellow, Mountain, Arid
- 💧 **Moisture Estimation** — Albedo-based (Gomez et al. 2008)
- 🧂 **Salinity Index** — Brightness × (1 − Saturation) method (Metternicht & Zinck 2003)
- 🌿 **Organic Matter Index** — Soil darkness correlation (Konen 2003, Viscarra Rossel 2012)
- ⚗️ **pH Estimation** — RGB colorimetry via HSV analysis (Viscarra Rossel 2006, Barron & Torrent 1986)
- 🏥 **Soil Health Score** — Multi-criterion evaluation against type-specific optimal ranges
- 📊 **Batch Processing** — All images run in a single batched model inference for speed
- 🗂️ **Upload History** — Session-based image history automatically saved
- 💬 **User Feedback** — Expert corrections logged to CSV for future model improvement
- 🖼️ **Before/After Preview** — Side-by-side comparison of original vs calibrated images

---

## Supported Soil Types

| Soil Type | pH Tendency | Key Characteristic |
|---|---|---|
| Alluvial Soil | Neutral to Slightly Acidic | Fertile, river-deposited |
| Black Soil | Neutral | High organic matter, moisture-retentive |
| Laterite Soil | Acidic | High iron content, less fertile |
| Red Soil | Acidic | Low OM, high iron oxide |
| Yellow Soil | Slightly Acidic | Moderate fertility, humid regions |
| Mountain Soil | Neutral to Acidic | Hilly terrain, low fertility |
| Arid Soil | Alkaline | Dry, sandy, low organic matter |

---

## Property Estimation — Research Basis

| Property | Formula | Source |
|---|---|---|
| **Brightness Index (BI)** | `√((R² + G² + B²) / 3)` | Mathieu et al. (1998) |
| **Redness Index (RI)** | `R² / (B × G)` | Escadafal (1989) |
| **Moisture (SMI)** | `(1 - albedo) × 100` | Gomez et al. (2008) |
| **Salinity (SI)** | `BI_norm × (1 - saturation) × 100` | Metternicht & Zinck (2003) |
| **OM Index** | `om_min + (om_max - om_min) × darkness` | Konen (2003), Viscarra Rossel & Webster (2012) |
| **pH** | HSV hue/saturation/value mapping | Viscarra Rossel et al. (2006), Barron & Torrent (1986) |

---

## Project Structure

```
tarp_project/
│
├── app.py                        # Gradio web application
├── requirements.txt
├── .gitignore
├── Block Diagram.svg
│
├── model/
│   ├── calibrator.py             # Self-calibrating AI: lighting detection + normalization
│   ├── predictor.py              # Feature engine + inference pipeline
│   ├── config.py                 # Paths and constants
│   ├── feedback.py               # User feedback CSV logger
│   └── soil_knowledge_base.json  # Per-type property ranges and pH
│
├── saved_model/                  # Trained EfficientNetV2 weights
├── data/                         # Runtime uploads and session history
└── Soil-Classification-Dataset/  # Training dataset (git-ignored)
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the repository
```bash
git clone <repo-url>
cd tarp_project
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**Dependencies:**
```
tensorflow
opencv-python
pillow
numpy
pandas
scikit-learn
matplotlib
seaborn
gradio
```

### 4. Run the application
```bash
python app.py
```

Open your browser at `http://127.0.0.1:7860`

---

## Usage

1. **Upload** one or more soil images using the file uploader
2. (Optional) Toggle **🔬 Enable Auto-Calibration** — on by default, normalizes lighting & color before classification
3. Click **Determine Soil Health**
4. View the **Final Decision** table — a Property/Value summary for the overall batch
5. Expand **Show Individual Image Results** to see per-image breakdowns
6. Expand **🔬 Calibration Report** to see detected lighting conditions, corrections applied, and before/after image comparisons
7. Optionally add a **user comment or correction** to provide feedback
8. Click **Reset** to clear all images and start a new session

---

## Output Fields

| Field | Description |
|---|---|
| **Soil Type** | Predicted soil class (majority vote across images) |
| **Confidence** | Weighted average prediction confidence |
| **Moisture** | Estimated soil moisture index (0–100 scale) |
| **Salinity** | Estimated electrical conductivity proxy (0–100 scale) |
| **OM Index** | Estimated organic matter index (%) |
| **pH Tendency** | Estimated pH value + qualitative label (e.g. `6.2 (Slightly Acidic)`) |
| **Soil Health** | Overall rating: Excellent / Good / Fair / Poor |

---

## Self-Calibrating AI — Automatic Calibration Layer

Mobile-captured soil images vary widely in lighting, white balance, and exposure. The **Self-Calibrating AI** layer automatically detects and corrects these issues before classification, ensuring consistent predictions regardless of capture conditions.

### Calibration Pipeline

| Step | Technique | Purpose |
|---|---|---|
| 1. **Lighting Detection** | Channel statistics (RGB mean, std) | Classify as overexposed / underexposed / warm cast / cool cast / low contrast / normal |
| 2. **Exposure Normalization** | CLAHE in LAB color space | Adaptive histogram equalization — stronger for extreme conditions, lighter for normal |
| 3. **White Balance Correction** | Gray World algorithm | Scales R/G/B channels toward neutral gray to remove color casts |

### Detected Conditions

| Condition | Trigger | Correction Strength |
|---|---|---|
| ☀️ Overexposed | Mean brightness > 200 | Strong CLAHE (clip=4.0) + moderate WB |
| 🌑 Underexposed | Mean brightness < 50 | Strong CLAHE (clip=4.0) + moderate WB |
| 🔶 Warm Cast | Red channel dominates by >25 | Standard CLAHE + heavy WB (α=0.9) |
| 🔷 Cool Cast | Blue channel dominates by >25 | Standard CLAHE + heavy WB (α=0.9) |
| 🌫️ Low Contrast | Pixel std-dev < 30 | Medium CLAHE (clip=3.5) + moderate WB |
| ✅ Normal | Within acceptable ranges | Light CLAHE (clip=2.0), no WB correction |

### Calibration Report

Each prediction includes a per-image calibration report showing:
- Detected lighting condition
- Corrections applied
- Brightness delta (change in mean brightness)
- Before/after image comparison gallery

---

## Novelty

1. **Self-Calibrating AI** — Automatic lighting detection and image normalization ensures consistent predictions across variable capture environments without manual adjustment
2. **Post-Classification Branching Pipeline** — Reuses EfficientNetV2's learned visual representations to drive downstream property estimation without retraining
3. **Hybrid Inference** — Combines deep learning classification with agronomic rule-based heuristics grounded in published spectral soil science
4. **No specialist hardware required** — Standard smartphone RGB camera is sufficient
5. **Explainability** — Property estimates are tied to interpretable, published formulas, not black-box predictions

---

## References

- Tan, M. & Le, Q. V. (2021). *EfficientNetV2: Smaller models and faster training.* ICML.
- Gomez, C. et al. (2008). *Evaluating the fit of soil reflectance models.* Geoderma.
- Metternicht, G. I. & Zinck, J. A. (2003). *Remote sensing of soil salinity: potentials and constraints.* Remote Sensing of Environment.
- Konen, M. E. et al. (2003). *Equations for predicting soil organic carbon using loss-on-ignition.* SSSAJ.
- Viscarra Rossel, R. A. & Webster, R. (2012). *Predicting soil properties from the Australian soil visible–near infrared spectroscopic database.* European Journal of Soil Science.
- Viscarra Rossel, R. A. et al. (2006). *Using data mining to model and interpret soil diffuse reflectance spectra.* Geoderma.
- Mathieu, R. et al. (1998). *Assessment of field reflectance with SPOT HRV data.* Remote Sensing of Environment.
- Escadafal, R. (1989). *Remote sensing of arid soil color with Landsat TM.* Advances in Space Research.
- Barron, V. & Torrent, J. (1986). *Use of the Kubelka-Munk theory to study the influence of iron oxides on soil colour.* European Journal of Soil Science.
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual explanations from deep networks.* ICCV.
- LeCun, Y., Bengio, Y. & Hinton, G. (2015). *Deep learning.* Nature.
