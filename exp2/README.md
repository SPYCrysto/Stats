# exp2 — Descriptive Statistics & Visual Analysis

Description
- Comprehensive descriptive statistics and visual analysis for the Pima Indians Diabetes dataset. Prints head/tail, summaries, mean/median/mode, range, variance, histograms, boxplots, pairplots, heatmap, and several scatterplots.

Dataset
- Expects `diabetes.csv` in the same folder (provided in repository).

Requirements
- Python 3.8+
- pandas, numpy, matplotlib, seaborn

Install

```powershell
python -m venv .venv
.\.venv\Scripts\pip install --upgrade pip
.\.venv\Scripts\pip install pandas numpy matplotlib seaborn
```

Run

```powershell
python exp2.py
```

Outputs
- Console statistics and interactive plots including histograms, boxplots, pairplot and correlation heatmap.

Notes
- The script contains an IQR-based outlier detection section and prints observations at the end.
