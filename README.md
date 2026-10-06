# Stats

All experiments performed in the lab.

This repository contains a collection of statistical analysis experiments and scripts built in Python. The work focuses on exploratory data analysis, descriptive statistics, and visual analytics using the Pima Indians Diabetes dataset.

## Repository structure

- `exp1/` — Exploratory data analysis (EDA)
- `exp2/` — Descriptive statistics and visual analysis
- `exp3/` through `exp9/` — Additional statistical experiments and analysis workflows

## Overview

Each experiment directory contains Python scripts and supporting documentation for a specific analysis task. Most scripts rely on the dataset file `diabetes.csv` placed in the same folder as the experiment.

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn

## Quick start

1. Clone the repository and open the project folder.

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the required dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install pandas numpy matplotlib seaborn
   ```

4. Run an experiment from its folder:

   ```bash
   cd exp1
   python exp1.py
   ```

   You can repeat the same pattern for other experiment folders, such as `exp2`, `exp3`, etc.

## Dataset

Each experiment expects the dataset file `diabetes.csv` to be available in its working directory. Refer to the experiment-specific READMEs for more details about the expected file placement and output.

## Notes

- This repository is intended for lab experiments and statistical learning exercises.
- For more detailed explanations, run instructions, and outputs, consult the README in each experiment folder.

## License

This project is distributed for educational and experimental use.
