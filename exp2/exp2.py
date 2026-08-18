import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

df = pd.read_csv("diabetes.csv")

print("=" * 60)
print("PIMA INDIANS DIABETES DATASET")
print("=" * 60)

print("\nFirst 5 Rows")
print(df.head())

print("\nLast 5 Rows")
print(df.tail())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns.tolist())

print("\nData Types")
print(df.dtypes)

print("\nDataset Information")
df.info()

print("\nMissing Values")
print(df.isnull().sum())

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print("\nSummary Statistics")
print(df.describe())

print("\nMean")
print(df.mean(numeric_only=True))

print("\nMedian")
print(df.median(numeric_only=True))

print("\nMode")
print(df.mode().iloc[0])

print("\nMinimum")
print(df.min(numeric_only=True))

print("\nMaximum")
print(df.max(numeric_only=True))

print("\nVariance")
print(df.var(numeric_only=True))

print("\nStandard Deviation")


print(df.std(numeric_only=True))

print("\nRange")
range_values = df.max(numeric_only=True) - df.min(numeric_only=True)
print(range_values)

df.hist(figsize=(15, 12), bins=20, edgecolor='black')
plt.suptitle("Histograms of Numerical Variables", fontsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 7))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title("Boxplots of Numerical Variables")
plt.show()

columns = [
"Pregnancies",
"Glucose",
"BloodPressure",
"SkinThickness",
"Insulin",
"BMI",
"DiabetesPedigreeFunction",
"Age"


]

for col in columns:
    plt.figure(figsize=(5, 5))
    sns.boxplot(y=df[col], color="skyblue")
    plt.title(f"Boxplot of {col}")
    plt.show()

plt.figure(figsize=(6, 5))
sns.countplot(x="Outcome", data=df)
plt.title("Frequency of Outcome")
plt.xlabel("Outcome (0 = Non-Diabetic, 1 = Diabetic)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(7, 5))
sns.scatterplot(
x="Glucose",
y="BMI",
hue="Outcome",
data=df
)

plt.title("Glucose vs BMI")
plt.show()


plt.figure(figsize=(7, 5))
sns.scatterplot(
x="Age",
y="Glucose",
hue="Outcome",
data=df
)

plt.title("Age vs Glucose")
plt.show()

pair_columns = [
"Glucose",
"BMI",
"Age",
"BloodPressure",
"Outcome"
]

sns.pairplot(
df[pair_columns],
hue="Outcome",
diag_kind="hist"
)

plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(
df.corr(),
annot=True,
cmap="coolwarm",
linewidths=0.5
)

plt.title("Correlation Matrix")
plt.show()

std_values = df.std(numeric_only=True)

highest_std_variable = std_values.idxmax()

print("\nVariable with Highest Standard Deviation")
print(highest_std_variable)
print("Standard Deviation =", std_values.max())

difference = abs(
df.mean(numeric_only=True) -
df.median(numeric_only=True)
)

print("\nDifference Between Mean and Median")


print(difference)

largest_difference = difference.idxmax()

print("\nVariable with Largest Mean-Median Difference")
print(largest_difference)

print("\nOutlier Detection using IQR")

for col in columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
(df[col] < lower_limit) |
(df[col] > upper_limit)
]

print(f"{col}: {len(outliers)} outliers")


print("\n" + "=" * 60)
print("OBSERVATIONS")
print("=" * 60)

print("1. Histograms show the distribution of numerical variables.")
print("2. Boxplots indicate outliers in Insulin, BMI, Pregnancies and DiabetesPedigreeFunction.")
print("3. Glucose has a positive relationship with diabetes outcome.")
print("4. Scatter plots show a weak positive relationship between Glucose and BMI.")
print("5. Correlation heatmap shows Glucose has one of the strongest correlations with Outcome.")

print("\nExperiment Completed Successfully.")