# exp5 — Resampling: Bootstrap & Permutation Tests

Description
- Demonstrates bootstrap confidence intervals for the mean Glucose value and a permutation test comparing mean Glucose between diabetic and non-diabetic groups. Produces plots of the bootstrap and permutation distributions.

Dataset
- This script downloads the dataset from a public URL; it does not require a local CSV.

Requirements
- Python 3.8+
- numpy, pandas, matplotlib, seaborn

Install

```powershell
python -m venv .venv
.\.venv\Scripts\pip install --upgrade pip
.\.venv\Scripts\pip install numpy pandas matplotlib seaborn
```

Run

```powershell
python exp5.py
```

Outputs
- Console summary of bootstrap and permutation test results, plus two interactive histograms.

Notes
- Random seeds are set for reproducibility within the script.
