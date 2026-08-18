import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv(url, names=columns)

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nDataset Information:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe())
sample_mean_glucose = df["Glucose"].mean()

print("\nSample Mean Glucose:", sample_mean_glucose)

diabetic_count = df["Outcome"].sum()
total_count = len(df)

sample_proportion = diabetic_count / total_count

print("Number of diabetic patients:", diabetic_count)
print("Total patients:", total_count)
print("Sample Proportion:", sample_proportion)
print("Sample Proportion (%):", sample_proportion * 100)

n = len(df)
mean = df["Glucose"].mean()
std = df["Glucose"].std()

confidence = 0.95
alpha = 1 - confidence


t_critical = stats.t.ppf(1 - alpha/2, df=n-1)

margin_error = t_critical * (std / np.sqrt(n))

lower_ci = mean - margin_error
upper_ci = mean + margin_error

print("\n95% Confidence Interval for Mean Glucose:")
print("Lower Limit:", lower_ci)
print("Upper Limit:", upper_ci)

p = sample_proportion

se_proportion = np.sqrt((p * (1 - p)) / n)

z_critical = stats.norm.ppf(1 - alpha/2)

lower_prop_ci = p - z_critical * se_proportion
upper_prop_ci = p + z_critical * se_proportion

print("\n95% Confidence Interval for Diabetes Proportion:")
print("Lower Limit:", lower_prop_ci)
print("Upper Limit:", upper_prop_ci)

reference_value = 120

t_statistic, p_value = stats.ttest_1samp(
    df["Glucose"],
    reference_value
)

print("\nOne-Sample t-Test")
print("H0: Mean glucose = 120")
print("H1: Mean glucose != 120")

print("t-statistic:", t_statistic)
print("p-value:", p_value)

alpha = 0.05

if p_value < alpha:
    print("Decision: Reject H0")
    print("Conclusion: There is significant evidence that the mean glucose level differs from 120 mg/dL.")
else:
    print("Decision: Fail to reject H0")
    print("Conclusion: There is insufficient evidence to conclude that the mean glucose level differs from 120 mg/dL.")


bmi_non_diabetic = df[df["Outcome"] == 0]["BMI"]
bmi_diabetic = df[df["Outcome"] == 1]["BMI"]

t_stat_bmi, p_value_bmi = stats.ttest_ind(
    bmi_diabetic,
    bmi_non_diabetic,
    equal_var=False
)

print("\nTwo-Sample t-Test for BMI")
print("H0: Mean BMI of diabetic and non-diabetic patients is equal")
print("H1: Mean BMI of diabetic and non-diabetic patients is different")

print("Mean BMI - Non-diabetic:", bmi_non_diabetic.mean())
print("Mean BMI - Diabetic:", bmi_diabetic.mean())

print("t-statistic:", t_stat_bmi)
print("p-value:", p_value_bmi)

if p_value_bmi < alpha:
    print("Decision: Reject H0")
    print("Conclusion: There is a significant difference in BMI between diabetic and non-diabetic patients.")
else:
    print("Decision: Fail to reject H0")
    print("Conclusion: There is insufficient evidence of a difference in BMI.")


df["BMI_Category"] = pd.cut(
    df["BMI"],
    bins=[0, 25, 30, np.inf],
    labels=["Normal", "Overweight", "Obese"]
)
contingency_table = pd.crosstab(
    df["BMI_Category"],
    df["Outcome"]
)

print("\nContingency Table:")
print(contingency_table)
chi2, p_chi, dof, expected = stats.chi2_contingency(
    contingency_table
)

print("\nChi-Square Test")
print("H0: BMI category and diabetes outcome are independent")
print("H1: BMI category and diabetes outcome are associated")

print("Chi-square statistic:", chi2)
print("Degrees of freedom:", dof)
print("p-value:", p_chi)

if p_chi < alpha:
    print("Decision: Reject H0")
    print("Conclusion: BMI category and diabetes outcome are significantly associated.")
else:
    print("Decision: Fail to reject H0")
    print("Conclusion: There is insufficient evidence of an association.")

sns.histplot(df["Glucose"], kde=True)
plt.axvline(120, linestyle="--", label="Reference = 120")
plt.title("Distribution of Glucose Levels")
plt.xlabel("Glucose")
plt.ylabel("Frequency")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x="Outcome", y="BMI", data=df)
plt.title("BMI Distribution by Diabetes Outcome")
plt.xlabel("Diabetes Outcome (0 = No, 1 = Yes)")
plt.ylabel("BMI")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="BMI_Category", hue="Outcome", data=df)
plt.title("BMI Category vs Diabetes Outcome")
plt.xlabel("BMI Category")
plt.ylabel("Number of Patients")
plt.show()