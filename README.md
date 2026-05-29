# Drosophila Addiction Tracking

Computational pipeline for analyzing *Drosophila melanogaster* locomotor behavior under chronic ethanol exposure. Built as part of a Summer Internship Program (SIP) research project.

## Overview

This project investigates locomotor sensitization and neural anchor effects in fruit flies across repeated ethanol exposures. Fly pose is tracked using [SLEAP](https://sleap.ai/), and a suite of kinematic features are extracted to characterize behavioral changes over a 48-hour period.

**Key findings visualized:**
- Chronic locomotor sensitization across two exposure days
- Neural anchor: individual jitter stabilization after repeated exposure
- Neural capture: egocentric gaze distribution

## Project Structure

```
├── src/
│   ├── preprocessing/      # Raw H5 → feature tensors
│   │   ├── preprocess_group1.py
│   │   ├── preprocess_group3.py
│   │   ├── build_tensor.py
│   │   ├── compress.py
│   │   ├── extract_features.py
│   │   ├── extract_trials.py
│   │   └── merge_csvs.py
│   ├── analysis/           # Statistical analysis & figure generation
│   │   ├── baseline.py
│   │   ├── sensitization.py
│   │   ├── process_data.py
│   │   ├── rainplot.py
│   │   ├── generatecsv.py
│   │   ├── generate_figures.py
│   │   └── validate_tensor.py
│   ├── modeling/           # SLEAP model inference & evaluation
│   │   ├── model.py
│   │   └── loss_curve.py
│   └── stimuli/            # Visual stimulus generation
│       ├── stimulus.py
│       └── looming.py
├── data/
│   ├── raw/                # Raw group CSVs from SLEAP export
│   └── processed/          # Analysis-ready outputs
├── figures/                # Publication-quality figures (600 DPI)
├── stimuli/                # HTML stimulus displays
└── requirements.txt
```

## Pipeline

1. **Pose tracking** — SLEAP labels `.slp` files (not included; see raw data)
2. **Preprocessing** — `src/preprocessing/preprocess_group*.py` extracts 5 kinematic features per fly per window: walking velocity, angular speed, spine angle, thigmotaxis, jitter
3. **Tensor compression** — `compress.py` and `build_tensor.py` produce `.npz` tensors (excluded from repo due to size; regenerate from raw H5 files)
4. **Analysis** — `src/analysis/` runs statistics (ANOVA, paired t-tests, Rayleigh test) and generates raincloud plots
5. **Figures** — `generate_figures.py` produces publication-ready figures at 600 DPI

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Raw H5 tracking data is stored externally (not included in repo). Contact for data access.

## Skills Demonstrated

- Pose estimation & animal behavior analysis (SLEAP)
- Feature engineering on time-series kinematics
- Statistical analysis: ANOVA, paired tests, circular statistics
- Data visualization: raincloud plots, polar histograms
- Python: NumPy, SciPy, Matplotlib, Seaborn, PyTorch, H5py
