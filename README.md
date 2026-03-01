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
   Final Report + Individual Image Results
```

---

## Features

- 🌱 **Soil Type Classification** — 7 classes: Alluvial, Black, Laterite, Red, Yellow, Mountain, Arid
- 💧 **Moisture Estimation** — Albedo-based (Gomez et al. 2008)
- 🧂 **Salinity Index** — Brightness × (1 − Saturation) method (Metternicht & Zinck 2003)
- 🌿 **Organic Matter Index** — Soil darkness correlation (Konen 2003, Viscarra Rossel 2012)
- ⚗️ **pH Estimation** — RGB colorimetry via HSV analysis (Viscarra Rossel 2006, Barron & Torrent 1986)
- 🏥 **Soil Health Score** — Multi-criterion evaluation against type-specific optimal ranges
- 📊 **Batch Processing** — All images run in a single batched model inference for speed
- 🗂️ **Upload History** — Session-based image history automatically saved
- 💬 **User Feedback** — Expert corrections logged to CSV for future model improvement

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
2. Click **Determine Soil Health**
3. View the **Final Decision** table — a Property/Value summary for the overall batch
4. Expand **Show Individual Image Results** to see per-image breakdowns
5. Optionally add a **user comment or correction** to provide feedback
6. Click **Reset** to clear all images and start a new session

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

## Novelty

1. **Post-Classification Branching Pipeline** — Reuses EfficientNetV2's learned visual representations to drive downstream property estimation without retraining
2. **Hybrid Inference** — Combines deep learning classification with agronomic rule-based heuristics grounded in published spectral soil science
3. **No specialist hardware required** — Standard smartphone RGB camera is sufficient
4. **Explainability** — Property estimates are tied to interpretable, published formulas, not black-box predictions

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
