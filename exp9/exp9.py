# ============================================================
# PIMA INDIANS DIABETES DATASET
# COMPLETE PYTHON SCRIPT FOR .py FILE
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.calibration import calibration_curve

import shap
from lime.lime_tabular import LimeTabularExplainer

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. LOAD DATASET
# ============================================================

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

df = pd.read_csv(url, names=columns)

print("=" * 70)
print("PIMA INDIANS DIABETES DATASET")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nClass Distribution:")
print(df["Outcome"].value_counts())

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 2. PREPROCESSING
# ============================================================

zero_as_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)

print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Set Size:", X_train.shape)
print("Testing Set Size:", X_test.shape)


# ============================================================
# 5. MACHINE LEARNING PIPELINE
# ============================================================

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])


# ============================================================
# 6. 5-FOLD CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "f1": "f1"
}

cv_results = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring
)

cv_accuracy = cv_results["test_accuracy"]
cv_f1 = cv_results["test_f1"]

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

print("\nAccuracy:")
for i, score in enumerate(cv_accuracy, 1):
    print("Fold", i, ":", round(score, 4))

print("\nF1 Score:")
for i, score in enumerate(cv_f1, 1):
    print("Fold", i, ":", round(score, 4))

print("\nMean Accuracy:", round(cv_accuracy.mean(), 4))
print("Accuracy Std:", round(cv_accuracy.std(), 4))

print("\nMean F1 Score:", round(cv_f1.mean(), 4))
print("F1 Std:", round(cv_f1.std(), 4))


# ============================================================
# 7. TRAIN FINAL MODEL
# ============================================================

pipeline.fit(X_train, y_train)


# ============================================================
# 8. TEST PREDICTIONS
# ============================================================

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]


# ============================================================
# 9. PERFORMANCE
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 70)
print("TEST SET PERFORMANCE")
print("=" * 70)

print("\nAccuracy:", round(accuracy, 4))
print("F1 Score:", round(f1, 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Non-diabetic", "Diabetic"]
))


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-diabetic", "Diabetic"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()


# ============================================================
# 11. BRIER SCORE
# ============================================================

brier = brier_score_loss(y_test, y_prob)

print("\n" + "=" * 70)
print("BRIER SCORE")
print("=" * 70)

print("\nBrier Score:", round(brier, 4))


# ============================================================
# 12. CALIBRATION CURVE
# ============================================================

prob_true, prob_pred = calibration_curve(
    y_test,
    y_prob,
    n_bins=10,
    strategy="uniform"
)

plt.figure(figsize=(7, 6))

plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    linewidth=2,
    label="Logistic Regression"
)

plt.plot(
    [0, 1],
    [0, 1],
    "--",
    color="black",
    label="Perfect Calibration"
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Frequency")
plt.title("Calibration Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# ============================================================
# 13. SHAP
# ============================================================

print("\n" + "=" * 70)
print("SHAP EXPLAINABILITY")
print("=" * 70)

imputer = pipeline.named_steps["imputer"]
scaler = pipeline.named_steps["scaler"]
model = pipeline.named_steps["classifier"]

X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)

X_train_scaled = scaler.transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

X_test_scaled_df = pd.DataFrame(
    X_test_scaled,
    columns=X.columns
)

explainer = shap.LinearExplainer(
    model,
    X_train_scaled
)

shap_values = explainer(X_test_scaled)


# ============================================================
# 14. SHAP SUMMARY
# ============================================================

shap.summary_plot(
    shap_values,
    X_test_scaled_df,
    show=False
)

plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.show()


# ============================================================
# 15. SHAP FEATURE IMPORTANCE
# ============================================================

mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Mean Absolute SHAP": mean_abs_shap
})

feature_importance = feature_importance.sort_values(
    "Mean Absolute SHAP",
    ascending=False
)

print("\nSHAP Feature Importance:")
print(feature_importance.to_string(index=False))

plt.figure(figsize=(8, 5))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Mean Absolute SHAP"],
    color="steelblue"
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title("SHAP Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


# ============================================================
# 16. INDIVIDUAL PATIENT
# ============================================================

sample_index = 0

print("\n" + "=" * 70)
print("INDIVIDUAL PATIENT PREDICTION")
print("=" * 70)

print("\nPatient Features:")
print(X_test.iloc[[sample_index]])

print("\nActual Outcome:",
      y_test.iloc[sample_index])

print("Predicted Outcome:",
      y_pred[sample_index])

print("Predicted Probability:",
      round(y_prob[sample_index], 4))


# ============================================================
# 17. SHAP WATERFALL
# ============================================================

print("\nGenerating SHAP individual explanation...")

shap.plots.waterfall(
    shap_values[sample_index],
    max_display=10
)

plt.show()


# ============================================================
# 18. LIME
# ============================================================

print("\n" + "=" * 70)
print("LIME EXPLAINABILITY")
print("=" * 70)

lime_explainer = LimeTabularExplainer(
    X_train_scaled,
    feature_names=X.columns.tolist(),
    class_names=["Non-diabetic", "Diabetic"],
    mode="classification",
    random_state=42
)

lime_exp = lime_explainer.explain_instance(
    X_test_scaled[sample_index],
    model.predict_proba,
    num_features=len(X.columns)
)

print("\nLIME Explanation:")

for feature, weight in lime_exp.as_list():
    print(
        feature,
        ":",
        round(weight, 4)
    )


# ============================================================
# 19. FINAL RESULTS
# ============================================================

results = pd.DataFrame({
    "Metric": [
        "5-Fold Mean Accuracy",
        "5-Fold Accuracy Std",
        "5-Fold Mean F1",
        "5-Fold F1 Std",
        "Test Accuracy",
        "Test F1",
        "Brier Score"
    ],

    "Value": [
        cv_accuracy.mean(),
        cv_accuracy.std(),
        cv_f1.mean(),
        cv_f1.std(),
        accuracy,
        f1,
        brier
    ]
})

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(results.to_string(index=False))


# ============================================================
# 20. TOP FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP 5 IMPORTANT FEATURES")
print("=" * 70)

for rank, (_, row) in enumerate(
    feature_importance.head(5).iterrows(),
    1
):
    print(
        rank,
        ".",
        row["Feature"],
        "-> SHAP:",
        round(row["Mean Absolute SHAP"], 4)
    )


# ============================================================
# 21. FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

print("""
The Pima Indians Diabetes Dataset was used to train a
Logistic Regression classification model.

The model was evaluated using 5-fold stratified
cross-validation, accuracy, F1-score, Brier Score,
and a calibration curve.

SHAP was used to identify the features that had the
greatest influence on model predictions.

LIME was used to provide a local explanation for
an individual patient prediction.

The experiment demonstrates that machine learning
evaluation should consider not only classification
performance but also probability calibration and
model explainability.
""")

print("\nExperiment completed successfully!")
