# Diabetes Prediction Using Supervised Classification

## Aim

To build supervised classification models for predicting diabetes outcomes using the **Pima Indians Diabetes Dataset** and evaluate their performance using appropriate classification metrics.

## Objectives

- Build supervised classification models to predict the diabetes outcome of patients.
- Evaluate and interpret classification model performance using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.

## Dataset

**Pima Indians Diabetes Dataset**

- Total records: **768**
- Features: **8 medical/patient attributes**
- Target variable: `Outcome`
  - `0` = Non-Diabetic
  - `1` = Diabetic

### Features

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

## Technologies Used

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- VS Code / Jupyter Notebook / Google Colab

## Classification Models

Three supervised classification models were implemented:

1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Decision Tree**

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Separated input features (`X`) and target variable (`y`).
3. Identified medically invalid zero values in:
   - Glucose
   - BloodPressure
   - SkinThickness
   - Insulin
   - BMI
4. Replaced invalid zero values with missing values (`NaN`).
5. Filled missing values using **median imputation**.
6. Divided the dataset into training and testing sets using an **80:20 stratified split**.
7. Applied **StandardScaler** for Logistic Regression and KNN.

## Model Performance

The models were evaluated using the test dataset.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7078 | 0.6000 | 0.5000 | 0.5455 | 0.8130 |
| K-Nearest Neighbors | 0.7532 | 0.6600 | 0.6111 | 0.6346 | 0.7899 |
| Decision Tree | 0.7597 | 0.6393 | 0.7222 | 0.6783 | 0.7622 |

## Confusion Matrices

### Logistic Regression

```text
[[82 18]
 [27 27]]
```

- True Negative (TN): 82
- False Positive (FP): 18
- False Negative (FN): 27
- True Positive (TP): 27

### K-Nearest Neighbors

```text
[[83 17]
 [21 33]]
```

- True Negative (TN): 83
- False Positive (FP): 17
- False Negative (FN): 21
- True Positive (TP): 33

### Decision Tree

```text
[[78 22]
 [15 39]]
```

- True Negative (TN): 78
- False Positive (FP): 22
- False Negative (FN): 15
- True Positive (TP): 39

## Results and Interpretation

The **Decision Tree** achieved the highest accuracy (**75.97%**) and the highest recall (**72.22%**) among the three models. It also produced the highest F1-score (**67.83%**).

However, **Logistic Regression** achieved the highest ROC-AUC (**0.813**), indicating the strongest overall ability among these models to distinguish between diabetic and non-diabetic patients based on the ROC curve.

For diabetes screening, **recall is particularly important** because false negatives represent diabetic patients who were incorrectly classified as non-diabetic.

## Evaluation Metrics

### Accuracy

Measures the proportion of correctly classified observations.

### Precision

Measures the proportion of predicted diabetic cases that were actually diabetic.

### Recall

Measures the proportion of actual diabetic cases that were correctly identified.

### F1-Score

Provides a balance between precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across different classification thresholds.

## Limitations

1. The dataset contains only **768 records**, so the models may not generalize perfectly to other populations.
2. Several medical attributes contain invalid zero values. Replacing these values through median imputation is an approximation and may affect predictions.
3. Model performance may change with a different train-test split or model hyperparameters.
4. These models are intended for educational machine-learning analysis and should not be treated as clinical diagnostic systems.

## Project Structure

```text
Diabetes-Classification/
│
├── diabetes.csv
├── diabetes_classification.py
├── README.md
└── screenshots/
    ├── model_comparison.png
    ├── confusion_matrix_logistic_regression.png
    ├── confusion_matrix_knn.png
    ├── confusion_matrix_decision_tree.png
    └── roc_curves.png
```

## How to Run

### 1. Install required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 2. Place the dataset

Keep `diabetes.csv` in the same directory as the Python program.

### 3. Run the program

```bash
python diabetes_classification.py
```

## Conclusion

Supervised classification models were successfully developed to predict diabetes outcomes using the Pima Indians Diabetes Dataset. Logistic Regression, KNN, and Decision Tree were evaluated using confusion matrices, accuracy, precision, recall, F1-score, and ROC-AUC.

The experiment demonstrated that different evaluation metrics provide different perspectives on model performance. The **Decision Tree** provided the best accuracy and recall in this experiment, while **Logistic Regression** achieved the highest ROC-AUC.

> **Note:** The reported results correspond to an 80:20 stratified split with `random_state=42` and the preprocessing used in the experiment.
