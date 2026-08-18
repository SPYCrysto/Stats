# ============================================================
# Experiment: Correlation, Distance Measures & Data Preprocessing
# Dataset: Pima Indians Diabetes Dataset
# ============================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances

# ------------------------------------------------------------
# Step 1: Load Dataset
# ------------------------------------------------------------
df = pd.read_csv("diabetes.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Information")
print(df.info())

# ------------------------------------------------------------
# Step 2: Identify Invalid Values
# ------------------------------------------------------------
# In the Pima dataset, zero values are invalid for these columns

invalid_cols = ["Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI"]

print("\nInvalid (Zero) Values")

for col in invalid_cols:
    print(f"{col}: {(df[col] == 0).sum()}")

# Replace invalid zeros with NaN

df[invalid_cols] = df[invalid_cols].replace(0, np.nan)

print("\nMissing Values After Replacing Zeros")
print(df.isnull().sum())

# ------------------------------------------------------------
# Step 3: Handle Missing Values using Median Imputation
# ------------------------------------------------------------

imputer = SimpleImputer(strategy="median")
df[invalid_cols] = imputer.fit_transform(df[invalid_cols])

print("\nMissing Values After Imputation")
print(df.isnull().sum())

# ------------------------------------------------------------
# Step 4: Correlation Matrix
# ------------------------------------------------------------

corr_matrix = df.corr()

print("\nCorrelation Matrix")
print(corr_matrix)

# ------------------------------------------------------------
# Step 5: Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix,
            annot=True,
            cmap="coolwarm",
            linewidths=0.5)

plt.title("Correlation Heatmap")
plt.show()

# ------------------------------------------------------------
# Step 6: Distance Measures
# Compare First Two Observations
# ------------------------------------------------------------

sample = df.iloc[[0,1], :-1]   # Ignore Outcome column

euclidean = euclidean_distances(sample)[0][1]
manhattan = manhattan_distances(sample)[0][1]
cosine = cosine_distances(sample)[0][1]

print("\nDistance Measures (Before Scaling)")
print("Euclidean Distance :", euclidean)
print("Manhattan Distance :", manhattan)
print("Cosine Distance    :", cosine)

# ------------------------------------------------------------
# Step 7: Standardization
# ------------------------------------------------------------

scaler = StandardScaler()

scaled_features = scaler.fit_transform(df.drop("Outcome", axis=1))

scaled_df = pd.DataFrame(scaled_features,
                         columns=df.columns[:-1])

scaled_df["Outcome"] = df["Outcome"]

print("\nFirst Five Rows Before Scaling")
print(df.head())

print("\nFirst Five Rows After Scaling")
print(scaled_df.head())

# ------------------------------------------------------------
# Step 8: Distance After Scaling
# ------------------------------------------------------------

sample_scaled = scaled_df.iloc[[0,1], :-1]

euclidean_scaled = euclidean_distances(sample_scaled)[0][1]
manhattan_scaled = manhattan_distances(sample_scaled)[0][1]
cosine_scaled = cosine_distances(sample_scaled)[0][1]

print("\nDistance Measures (After Scaling)")
print("Euclidean Distance :", euclidean_scaled)
print("Manhattan Distance :", manhattan_scaled)
print("Cosine Distance    :", cosine_scaled)

# ------------------------------------------------------------
# Step 9: Strongest Positive & Negative Correlation
# ------------------------------------------------------------

corr = corr_matrix.copy()

np.fill_diagonal(corr.values, np.nan)

positive = corr.unstack().dropna().sort_values(ascending=False)
negative = corr.unstack().dropna().sort_values()

print("\nStrongest Positive Correlation")
print(positive.head(1))

print("\nStrongest Negative Correlation")
print(negative.head(1))