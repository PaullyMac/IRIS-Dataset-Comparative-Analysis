# IRIS Dataset Comparative Analysis

This project provides a comprehensive comparative analysis of different decision tree algorithms on the classic IRIS dataset. It examines the performance, characteristics, and trade-offs between CART, ID3, and C5.0 decision tree implementations.

## Project Structure

IRIS Dataset (Comparative Analysis)  
├── .venv/                      # Virtual environment directory  
├── Decision Trees/             # Decision tree implementation and comparison  
│   ├── data/                   # Dataset directory  
│   │   └── Iris.csv            # Iris dataset CSV file  
│   ├── utils/                  # Utility functions  
│   │   └── visualization.py    # Visualization helper functions  
│   └── decision_tree_comparison.ipynb  # Main analysis notebook  
├── KNN-iris/                   # PCA and KNN analysis submodule  
│   ├── datasets/               # Data for PCA and KNN  
│   │   ├── Iris.csv            # Iris dataset CSV file  
│   │   ├── train_and_test2.csv # Pre-split train/test CSV  
│   │   └── database.sqlite     # SQLite database file  
│   ├── PCA.ipynb               # PCA analysis notebook  
│   ├── pca_aligned.png         # PCA alignment visualization  
│   ├── iris_pca_plot.png       # PCA scatter plot  
│   └── knn.ipynb               # KNN classification notebook  
└── README.md                   # This file  

## Overview

This project analyzes the classic Iris flower dataset using three different decision tree algorithms. The analysis compares their performance, structure, and characteristics to understand the strengths and weaknesses of each approach.

## Algorithms Compared

1. **CART (Classification and Regression Trees)** - Uses Gini impurity as splitting criterion
2. **ID3 (Iterative Dichotomiser 3)** - Uses entropy and information gain
3. **C5.0** - An improved version of ID3 with pruning and other enhancements

## Code Outline

### 1) Decision Trees (`Decision Trees/decision_tree_comparison.ipynb`)
1. Setup & environment checks (install packages, Graphviz note)  
2. Data loading (local CSV fallback to sklearn)  
3. Data exploration  
   - Descriptive statistics  
   - Missing‐value check  
   - Class distribution plot  
   - Feature histograms & pairplot  
   - Correlation heatmap  
4. Data preparation (train/test split, stratification)  
5. Model implementation & evaluation  
   - CART (Gini), ID3 (Entropy), C5.0 (Entropy + pruning)  
   - Accuracy, classification report, confusion matrix  
6. Tree visualization (Graphviz & matplotlib fallback)  
7. Decision boundary plotting  
8. Cross‑validation comparison (10‑fold)  
9. Tree complexity analysis (depth, nodes, leaves)  
10. Feature importance comparison  
11. Decision boundary comparison across models  
12. Confusion matrix comparison side‑by‑side  
13. Learning curve analysis  
14. Hyperparameter analysis  
    - Max depth  
    - Min samples split  
15. Final comparison table & conclusions  

### 2) PCA (`KNN-iris/PCA.ipynb`)
1. Import libraries & load `datasets/Iris.csv`  
2. Scatter plots of all feature pairs  
3. Manual covariance matrix computation  
4. Covariance heatmap (seaborn)  
5. Eigen decomposition & manual PCA projection  
6. Plot PCA projection with eigenvector arrows  
7. Sign‐alignment to match sklearn’s PCA  
8. Compare manual PCA vs `sklearn.decomposition.PCA`  

### 3) KNN (`KNN-iris/knn.ipynb`)
1. Define `euclidean_distance` & `KNearestNeighbors` class  
2. Load dataset & drop `Id` column  
3. Train/test split (scikit‑learn)  
4. Group training samples by species into a dict  
5. Fit KNN & predict on test set  
6. Evaluate performance  
   - Accuracy score  
   - Confusion matrix  
   - Classification report  
7. Plot confusion matrix heatmap  

## Utility Functions

The `visualization.py` module in the `utils` directory provides helper functions for:

- Plotting decision boundaries
- Visualizing decision tree structures
- Creating feature importance plots
- Generating confusion matrix visualizations

## Dataset

The classic Iris dataset consists of 150 samples from three species of Iris flowers:

- Setosa
- Versicolor
- Virginica

Each sample has four features:

- Sepal length
- Sepal width
- Petal length
- Petal width

## Results and Findings

The analysis reveals that:

- CART achieves the highest test accuracy (93.3%)
- C5.0 creates the simplest tree structure and shows the most stable cross-validation performance
- ID3 creates the most complex tree with intermediate performance
- Petal dimensions are far more important than sepal dimensions for classification
- All three algorithms perfectly separate the Setosa class but show varying degrees of confusion between Versicolor and Virginica

## Usage

To run the decision tree analysis, open  
[`Decision Trees/decision_tree_comparison.ipynb`](Decision%20Trees/decision_tree_comparison.ipynb).  

For PCA and KNN workflows:  
- Open [`KNN-iris/PCA.ipynb`](KNN-iris/PCA.ipynb) for PCA analysis.  
- Open [`KNN-iris/knn.ipynb`](KNN-iris/knn.ipynb) for KNN classification.  

Requirements:  
- Python 3.6+  
- Packages: numpy, pandas, matplotlib, seaborn, scikit-learn, graphviz  

Note: For tree visualizations to work properly, the Graphviz software needs to be installed separately on your system.
