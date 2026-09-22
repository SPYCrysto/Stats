# Experiment 6: Regression Models Using Statistical Performance Measures

## Course Information

-   **Course:** Statistics for Machine Learning and Data Science Lab
-   **Course Code:** 2015111
-   **Academic Year:** 2026--27
-   **Class:** TE-AI&DS
-   **Semester:** V
-   **Experiment Number:** 6
-   **Student Name:** Vedant Kishor Mhatre
-   **Roll No.:** 72

## Title

**Develop and Evaluate Regression Models using Statistical Performance
Measures**

## Aim

To develop regression models for predicting a continuous variable using
the Pima Indians Diabetes Dataset and evaluate their performance using
appropriate statistical metrics.

## Objectives

After completing this experiment, the student will be able to:

1.  Develop regression models to predict a continuous health-related
    variable from medical attributes.
2.  Evaluate and interpret the performance of regression models using
    appropriate statistical measures.

## Tools Required

-   Python 3.x
-   Google Colab / Jupyter Notebook
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn

## Dataset Details

The **Pima Indians Diabetes Dataset** contains medical information for
768 female patients.

In this experiment:

-   **Target variable:** BMI
-   **Input features:** Pregnancies, Glucose, BloodPressure,
    SkinThickness, Insulin, DiabetesPedigreeFunction, Age, and Outcome
-   **Task:** Predict the continuous BMI value using linear regression

## Procedure

1.  Load the Pima Indians Diabetes Dataset.
2.  Select a continuous target variable and relevant predictor
    variables.
3.  Check and preprocess missing or invalid values.
4.  Divide the dataset into training and testing sets.
5.  Develop a suitable regression model.
6.  Train the model using the training data.
7.  Generate predictions for the test data.
8.  Evaluate the model using appropriate performance metrics.
9.  Analyze residuals and compare actual and predicted values.
10. Interpret the model results and identify possible limitations.

## Statistical Methods Used

### 1. Linear Regression

Linear regression is a supervised learning technique used to predict a
continuous numerical output.

For multiple predictors:

\[ y = `\beta`{=tex}\_0 + `\beta`{=tex}\_1x_1 + `\beta`{=tex}\_2x_2 +
`\cdots `{=tex}+ `\beta`{=tex}\_nx_n + `\epsilon`{=tex} \]

Where:

-   `y` = target variable
-   `x` = predictor variables
-   `β` = model coefficients
-   `ε` = error term

### 2. Mean Absolute Error (MAE)

MAE represents the average absolute prediction error.

\[ MAE =
`\frac{1}{n}`{=tex}`\sum`{=tex}\_{i=1}\^{n}\|y_i-`\hat{y}`{=tex}\_i\| \]

A lower MAE indicates smaller average prediction errors.

### 3. Mean Squared Error (MSE)

MSE represents the average squared prediction error.

\[ MSE =
`\frac{1}{n}`{=tex}`\sum`{=tex}\_{i=1}\^{n}(y_i-`\hat{y}`{=tex}\_i)\^2
\]

Larger errors are penalized more heavily.

### 4. Root Mean Squared Error (RMSE)

\[ RMSE = `\sqrt{MSE}`{=tex} \]

RMSE is expressed in the same units as the target variable.

### 5. Coefficient of Determination (R²)

R² indicates the proportion of variation in the target variable
explained by the model.

### 6. Residual Analysis

A residual is calculated as:

\[ e = y-`\hat{y}`{=tex} \]

Residual analysis helps identify:

-   Systematic prediction errors
-   Nonlinear patterns
-   Unequal variance
-   Potential outliers

## Implementation

The Python implementation:

1.  Loads the dataset from GitHub.
2.  Replaces invalid zero values in selected medical columns with
    missing values.
3.  Performs median imputation.
4.  Selects BMI as the target variable.
5.  Splits the dataset into training and testing sets.
6.  Trains a Linear Regression model.
7.  Generates BMI predictions.
8.  Calculates MAE, MSE, RMSE, and R².
9.  Performs residual analysis.
10. Displays visualizations and the regression equation.

## How to Run

### Option 1: Google Colab

1.  Open [Google Colab](https://colab.research.google.com/).
2.  Create a new notebook.
3.  Copy the Python code into a code cell.
4.  Run the cell.
5.  View the printed results and generated plots.

### Option 2: Jupyter Notebook

1.  Install the required libraries:

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

2.  Open Jupyter Notebook.
3.  Create a new Python notebook.
4.  Paste the code.
5.  Run all cells.

## Performance Metrics

The model calculates the following metrics:

  Metric     Description
  ---------- ----------------------------------------------------
  MAE        Average absolute prediction error in BMI units
  MSE        Average squared prediction error
  RMSE       Typical prediction error in BMI units
  R² Score   Proportion of BMI variation explained by the model

The exact metric values are generated when the Python code is executed.

## Visualizations

The implementation generates the following visualizations:

1.  **Actual vs Predicted BMI Plot**\
    Compares actual BMI values with predicted BMI values.

2.  **Residual Plot**\
    Displays residuals against predicted BMI values.

3.  **Distribution of Residuals**\
    Shows the distribution of prediction errors using a histogram and
    KDE curve.

4.  **Correlation Matrix**\
    Displays correlations among the dataset variables using a heatmap.

## Interpretation

-   MAE indicates the average prediction error in BMI units.
-   MSE gives greater importance to larger errors.
-   RMSE represents the typical prediction error in BMI units.
-   R² indicates how much variation in BMI is explained by the model.
-   Residuals should ideally be randomly distributed around zero.
-   Points closer to the diagonal line in the actual-versus-predicted
    plot indicate more accurate predictions.

## Conclusion

A regression model was developed to predict a continuous health-related
variable using the Pima Indians Diabetes Dataset. The model was
evaluated using appropriate regression metrics, and residual analysis
was performed to understand prediction errors and possible limitations.

## Questions and Answers

### 1. Which variable was selected as the target, and why was it suitable for regression?

BMI (Body Mass Index) was selected as the target variable.

BMI is suitable because it is a continuous numerical variable, unlike
the Outcome variable, which is binary (0 or 1). Regression models are
designed to predict continuous numerical values.

The other medical attributes, such as Glucose, Blood Pressure, Insulin,
Age, and Pregnancies, were used as predictor variables.

### 2. What are the values of MAE, MSE, RMSE, and R² obtained by the model?

The values are calculated and printed when the Python program is
executed. They depend on the dataset, preprocessing, and train-test
split used in the implementation.

### 3. Which performance metric provides the most useful interpretation for this problem? Justify your answer.

MAE provides a useful interpretation for this problem because it
directly indicates the average prediction error in BMI units.

For example, if MAE is approximately 4.9, the model's predictions are,
on average, about 4.9 BMI units away from the actual BMI.

Although R² is useful for understanding how much variation is explained
by the model, MAE is easier to interpret because it is expressed in the
same units as BMI.

### 4. What patterns or issues were observed from the residual analysis?

Possible observations include:

-   Residuals are generally distributed around zero.
-   Some residuals may be relatively large.
-   The residual plot may show spread or variation rather than a
    perfectly uniform distribution.
-   The relationship between the medical variables and BMI may not be
    completely linear.
-   Possible outliers may be present.

Therefore, the linear regression model may provide useful predictions
but may not explain all variation in BMI.

### 5. State two limitations of the developed regression model.

#### Limitation 1: Limited Generalizability

The dataset contains 768 female patients from a specific population.
Therefore, the model may not perform equally well for other populations,
including males or different ethnic groups.

#### Limitation 2: Linear Regression Assumptions

The model assumes a primarily linear relationship between the predictors
and BMI. Real-world relationships between medical attributes and BMI may
be nonlinear, so a linear model may not capture all important patterns.

## Author

**Vedant Kishor Mhatre**\
Roll No.: 72
