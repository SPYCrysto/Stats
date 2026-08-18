# exp4 — Statistical Tests & Confidence Intervals

Description
- Performs inferential statistics on the Pima Indians Diabetes dataset: computes descriptive stats, confidence intervals for mean glucose and diabetes proportion, one-sample t-test for glucose against a reference value, two-sample t-test for BMI between groups, and a chi-square test for BMI category vs diabetes outcome. Also displays relevant plots.

Dataset
- This script downloads the dataset from a public URL; it does not require a local CSV.

Requirements
- Python 3.8+
- pandas, numpy, scipy, matplotlib, seaborn

Install

```powershell
python -m venv .venv
.\.venv\Scripts\pip install --upgrade pip
.\.venv\Scripts\pip install pandas numpy scipy matplotlib seaborn
```

Run

```powershell
python exp4.py
```

Outputs
- Console test statistics and figures (histogram, boxplot, countplot) summarizing inference results.

Notes
- Uses `scipy.stats` for t-tests, z-critical, and chi-square tests. Internet connection required for dataset download.
