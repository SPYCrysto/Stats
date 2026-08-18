# Stats

Repository of statistical experiments and analyses performed in the lab.

## Overview

This repository contains small experiments and example scripts that demonstrate exploratory data analysis (EDA) and descriptive statistical analyses on the Pima Indians Diabetes dataset.

## Experiments

- exp1 — Exploratory Data Analysis (EDA). See `exp1/README.md` for details.
- exp2 — Descriptive Statistics & Visual Analysis. See `exp2/README.md` for details.

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn

## Quick start

1. Create a virtual environment:

   python -m venv .venv

2. Activate it and install dependencies (Windows example):

   .\.venv\Scripts\pip install --upgrade pip
   .\.venv\Scripts\pip install pandas numpy matplotlib seaborn

   On macOS/Linux use `source .venv/bin/activate` and `pip install ...`.

3. Run an experiment (from the experiment folder):

   cd exp1
   python exp1.py

   or

   cd exp2
   python exp2.py

## Dataset

Place `diabetes.csv` in the same folder as the experiment you want to run (e.g., `exp1/` or `exp2/`). See each experiment's README for dataset details.

## Notes

See the per-experiment READMEs (`exp1/README.md`, `exp2/README.md`) for full descriptions, requirements, and example outputs.
