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
└── README.md                   # This file

## Overview

This project analyzes the classic Iris flower dataset using three different decision tree algorithms. The analysis compares their performance, structure, and characteristics to understand the strengths and weaknesses of each approach.

## Algorithms Compared

1. **CART (Classification and Regression Trees)** - Uses Gini impurity as splitting criterion
2. **ID3 (Iterative Dichotomiser 3)** - Uses entropy and information gain
3. **C5.0** - An improved version of ID3 with pruning and other enhancements

## Code Outline

The `decision_tree_comparison.ipynb` notebook contains the following major sections:

1. **Setup and Data Loading**
   - Installs and imports required libraries
   - Loads the Iris dataset either from local CSV or from scikit-learn
   - Sets up feature and target variables

2. **Data Exploration and Visualization**
   - Calculates descriptive statistics
   - Checks for missing values
   - Analyzes class distribution
   - Creates histograms of feature distributions by species
   - Generates pairplots showing relationships between features
   - Creates correlation matrix heatmap

3. **Data Preparation**
   - Splits dataset into training (70%) and test (30%) sets
   - Ensures class balance is maintained through stratification

4. **Model Implementation and Evaluation**
   - Implements CART, ID3, and C5.0 decision tree algorithms
   - Evaluates each model's performance (accuracy, classification report, confusion matrix)
   - Visualizes tree structures for each algorithm
   - Plots decision boundaries in feature space

5. **Cross-Validation Comparison**
   - Performs 10-fold cross-validation for robust performance comparison
   - Visualizes cross-validation results with boxplots and bar charts

6. **Tree Complexity Comparison**
   - Analyzes structural complexity metrics (depth, nodes, leaves)
   - Visualizes complexity differences between algorithms

7. **Feature Importance Comparison**
   - Compares how each algorithm ranks feature importance
   - Creates individual and comparative feature importance visualizations

8. **Decision Boundaries Comparison**
   - Side-by-side visualization of decision boundaries in feature space
   - Analyzes differences in how algorithms partition the data

9. **Confusion Matrix Comparison**
   - Visualizes classification errors for each model
   - Analyzes patterns of misclassification

10. **Learning Curve Analysis**
    - Generates learning curves for each algorithm
    - Evaluates how performance changes with increasing training data
    - Identifies potential overfitting or underfitting

11. **Hyperparameter Analysis**
    - Evaluates how maximum tree depth affects performance
    - Analyzes the impact of minimum samples split parameter
    - Determines optimal hyperparameter values

12. **Comparison Summary and Conclusions**
    - Comprehensive comparison table of all metrics
    - Summary of algorithm characteristics and trade-offs
    - Recommendations for different use cases

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

To run the analysis, open the `decision_tree_comparison.ipynb` notebook in Jupyter and execute the cells sequentially.

Requirements:
- Python 3.6+
- Required packages: numpy, pandas, matplotlib, seaborn, scikit-learn, graphviz

Note: For tree visualizations to work properly, the Graphviz software needs to be installed separately on your system.
