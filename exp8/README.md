# Experiment 8: Clustering and Dimensionality Reduction

## Aim

To apply clustering and dimensionality reduction techniques to the
**Pima Indians Diabetes Dataset** to discover hidden patterns and
visualize relationships among patients.

## Objectives

-   Apply clustering and dimensionality reduction techniques to discover
    patterns in a high-dimensional dataset.
-   Interpret the resulting clusters and reduced-dimensional
    representations using suitable evaluation measures and
    visualizations.

## Dataset

**Pima Indians Diabetes Dataset**

The dataset contains medical information for **768 female patients**.
The `Outcome` variable indicates diabetes status.

For unsupervised learning, the `Outcome` variable is excluded while
developing the clusters and is used later to compare the discovered
clusters with the known diabetes categories.

## Technologies and Libraries

-   Python 3.x
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Google Colab / Jupyter Notebook / VS Code

## Methods Used

### 1. Data Preprocessing

The dataset is loaded and inspected for:

-   Dataset shape and first few records
-   Data types and statistical summary
-   Missing values
-   Invalid zero values in medical attributes

Zero values in relevant medical columns are treated as missing values
and replaced using the median of the respective column.

### 2. Feature Selection

The following numerical features are used for clustering:

-   Pregnancies
-   Glucose
-   BloodPressure
-   SkinThickness
-   Insulin
-   BMI
-   DiabetesPedigreeFunction
-   Age

The `Outcome` column is not used as an input to the unsupervised
clustering model.

### 3. Feature Standardization

`StandardScaler` is used to standardize the selected features so that
variables with different numerical scales can be compared fairly during
clustering.

### 4. K-Means Clustering

K-Means clustering is applied to group patients with similar feature
patterns.

Different values of `k` from **2 to 10** are evaluated.

### 5. Elbow Method

The Elbow Method is used to examine the **inertia / Within-Cluster Sum
of Squares (WCSS)** for different numbers of clusters.

The selected value of `k` can be identified from the point where adding
additional clusters gives diminishing improvement.

### 6. Silhouette Score

The Silhouette Score is calculated for different values of `k` to
evaluate cluster separation and cohesion.

The experiment selects the `k` value corresponding to the highest
Silhouette Score.

### 7. Cluster Analysis

After applying K-Means, the experiment analyzes:

-   Cluster sizes
-   Mean feature values for each cluster
-   Standardized cluster centers
-   Cluster-wise diabetes outcome distribution

### 8. Principal Component Analysis (PCA)

PCA is applied with **2 principal components** to reduce the
dimensionality of the standardized dataset.

The experiment calculates the explained variance of:

-   PC1
-   PC2
-   Total variance explained by PC1 and PC2

### 9. Visualization

The experiment produces visualizations including:

-   Elbow Method curve
-   Silhouette Score plot
-   Cluster mean heatmap
-   Standardized cluster center heatmap
-   Cluster vs Outcome distribution
-   PCA explained variance plot
-   K-Means clusters in 2D PCA space
-   PCA representation colored by diabetes Outcome
-   PCA feature loading heatmap

## Experiment Workflow

``` text
Pima Indians Diabetes Dataset
            ↓
     Data Inspection
            ↓
 Handle Invalid/Missing Values
            ↓
     Feature Selection
            ↓
    Feature Standardization
            ↓
       K-Means Clustering
            ↓
      Elbow Method
            ↓
      Silhouette Score
            ↓
      Cluster Analysis
            ↓
          PCA
            ↓
      2D Visualization
            ↓
   Interpretation of Results
```

## Expected Results

The program reports:

-   Original dataset shape
-   Number of features used
-   Silhouette Scores for different values of `k`
-   Selected number of clusters
-   Best Silhouette Score
-   Cluster sizes
-   Mean feature values for each cluster
-   Cluster vs Outcome distribution
-   PCA explained variance for PC1 and PC2
-   Total variance explained by the two principal components
-   PCA feature loadings

The exact numerical results depend on the dataset and the execution of
the provided code.

## Key Concepts

### K-Means Clustering

K-Means divides observations into `k` clusters by minimizing the
distance between observations and their assigned cluster centroid.

### Elbow Method

The Elbow Method evaluates different values of `k` using within-cluster
sum of squares and helps identify a point where additional clusters
provide diminishing improvement.

### Silhouette Score

The Silhouette Score evaluates how well an observation fits within its
assigned cluster compared with the nearest other cluster. A higher value
generally indicates better-defined clusters.

### PCA

Principal Component Analysis transforms correlated variables into a
smaller number of uncorrelated principal components. The first principal
components capture as much variance as possible and can be used to
visualize high-dimensional data in two dimensions.

## How to Run

1.  Install Python 3.x.
2.  Install the required libraries:

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

3.  Place `diabetes.csv` in the project directory.
4.  Run the Python program or notebook.
5.  Review the printed results and generated plots.

## Project Structure

``` text
Experiment-8/
│
├── diabetes.csv
├── exp8.py
└── README.md
```

## Conclusion

This experiment applies **K-Means clustering** and **PCA** to the Pima
Indians Diabetes Dataset. Clustering is used to discover groups of
patients with similar characteristics without using the diabetes outcome
during model development. PCA reduces the feature space to two
dimensions, making it possible to visualize the discovered patterns and
relationships in the dataset.
