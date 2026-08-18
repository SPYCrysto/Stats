import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
variable = "Glucose"

data = df[variable].dropna().values

original_mean = np.mean(data)

print("\nSelected variable:", variable)
print("Original sample mean:", original_mean)

np.random.seed(42)

n_bootstrap = 10000
n = len(data)

bootstrap_means = np.empty(n_bootstrap)

for i in range(n_bootstrap):
    sample = np.random.choice(data, size=n, replace=True)
    bootstrap_means[i] = np.mean(sample)


lower = np.percentile(bootstrap_means, 2.5)
upper = np.percentile(bootstrap_means, 97.5)

print("\nBootstrap Results")
print("-----------------")
print("Original sample mean:", original_mean)
print("Bootstrap mean:", np.mean(bootstrap_means))
print("95% Bootstrap CI:", (lower, upper))


plt.figure(figsize=(10, 6))

sns.histplot(
    bootstrap_means,
    bins=50,
    kde=True,
    color="skyblue"
)

plt.axvline(
    original_mean,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Original Mean = {original_mean:.2f}"
)

plt.axvline(
    lower,
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"2.5th Percentile = {lower:.2f}"
)

plt.axvline(
    upper,
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"97.5th Percentile = {upper:.2f}"
)

plt.title("Bootstrap Distribution of Mean Glucose")
plt.xlabel("Bootstrap Sample Mean")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()


diabetic = df[df["Outcome"] == 1]["Glucose"].dropna()
non_diabetic = df[df["Outcome"] == 0]["Glucose"].dropna()

print("\nGroup Sizes")
print("-----------")
print("Diabetic:", len(diabetic))
print("Non-diabetic:", len(non_diabetic))


diabetic_mean = diabetic.mean()
non_diabetic_mean = non_diabetic.mean()

observed_difference = diabetic_mean - non_diabetic_mean

print("\nGroup Mean Results")
print("------------------")
print("Mean Glucose - Diabetic:", diabetic_mean)
print("Mean Glucose - Non-diabetic:", non_diabetic_mean)
print("Observed difference:", observed_difference)


np.random.seed(42)

n_permutations = 10000

combined = np.concatenate([
    diabetic.values,
    non_diabetic.values
])

n_diabetic = len(diabetic)

permutation_differences = np.empty(n_permutations)

for i in range(n_permutations):

    
    shuffled = np.random.permutation(combined)

    
    perm_diabetic = shuffled[:n_diabetic]
    perm_non_diabetic = shuffled[n_diabetic:]

    
    permutation_differences[i] = (
        np.mean(perm_diabetic)
        - np.mean(perm_non_diabetic)
    )


p_value = np.mean(
    np.abs(permutation_differences)
    >= np.abs(observed_difference)
)

print("\nPermutation Test Results")
print("------------------------")
print("Observed difference:", observed_difference)
print("Permutation p-value:", p_value)



plt.figure(figsize=(10, 6))

sns.histplot(
    permutation_differences,
    bins=50,
    kde=True,
    color="lightcoral"
)

plt.axvline(
    observed_difference,
    color="blue",
    linestyle="--",
    linewidth=2,
    label=f"Observed Difference = {observed_difference:.2f}"
)

plt.axvline(
    -observed_difference,
    color="blue",
    linestyle="--",
    linewidth=2
)

plt.title("Permutation Distribution of Difference in Mean Glucose")
plt.xlabel("Difference in Mean Glucose")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

alpha = 0.05

print("\nStatistical Decision")
print("--------------------")

if p_value < alpha:
    print("Reject the null hypothesis.")
    print("There is sufficient statistical evidence of a difference")
    print("in mean Glucose between diabetic and non-diabetic groups.")
else:
    print("Fail to reject the null hypothesis.")
    print("There is insufficient statistical evidence of a difference")
    print("in mean Glucose between the two groups.")
