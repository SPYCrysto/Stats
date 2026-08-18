# exp1 — Exploratory Data Analysis (EDA)

Description
- Basic exploratory data analysis on the Pima Indians Diabetes dataset. Prints dataset shape, types, descriptive statistics, missing/zero counts, and shows histograms, boxplots, countplot, correlation heatmap, and scatterplots.

Dataset
- Expects `diabetes.csv` in the same folder (provided in repository).

Requirements
- Python 3.8+
- pandas, matplotlib, seaborn

Install

```powershell
python -m venv .venv
.\.venv\Scripts\pip install --upgrade pip
.\.venv\Scripts\pip install pandas matplotlib seaborn
```

Run

```powershell
python exp1.py
```

Outputs
- Console summary statistics and figures displayed interactively (histograms, boxplots, heatmap, scatterplots).

Notes
- Zero values in some columns are reported but not imputed in this script.
