import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("diabetes.csv")

print("Shape of Dataset:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 Records:")
print(df.head())


print("\nNumerical Variables:")
for col in df.columns:
    if df[col].nunique() > 2:
        print("-", col)

print("\nBinary Variable:")
print("Outcome")


print("\nStatistical Summary")
print(df.describe())

print("\nMedian")
print(df.median())

print("\nVariance")
print(df.var())

print("\nStandard Deviation")
print(df.std())

print("\nMissing Values")
print(df.isnull().sum())

print("\nZero Values in Important Columns")

columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in columns:
    print(col, ":", (df[col] == 0).sum())

print("\nPossible Outliers (IQR Method)")

for col in df.columns[:-1]:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(col, ":", len(outliers))

df.hist(figsize=(12,10))
plt.show()

plt.figure(figsize=(12,8))

for i, col in enumerate(df.columns[:-1]):
    plt.subplot(3,3,i+1)
    sns.boxplot(y=df[col])

plt.tight_layout()
plt.show()

sns.countplot(x="Outcome", data=df)
plt.title("Diabetes Outcome")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

sns.scatterplot(data=df, x="Glucose", y="BMI", hue="Outcome")
plt.show()
