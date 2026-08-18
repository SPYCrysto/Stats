# exp3 — Correlation, Distance Measures & Preprocessing

Description
- Loads the Pima Indians Diabetes dataset, replaces invalid zeros with NaN, imputes missing values using median, computes correlation matrix, shows a heatmap, compares Euclidean/Manhattan/Cosine distances before and after standardization, and reports strongest correlations.

Dataset
- Expects `diabetes.csv` in the same folder (provided in repository).

Requirements
- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn

Install

```powershell
python -m venv .venv
.\.venv\Scripts\pip install --upgrade pip
.\.venv\Scripts\pip install pandas numpy matplotlib seaborn scikit-learn
```

Run

```powershell
python exp3.py
```

Outputs
- Console logs for imputation and distance measures, plus heatmap and other plots.

Notes
- The script uses `SimpleImputer(strategy='median')` and `StandardScaler` from scikit-learn.
