import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
import numpy as np
import pandas as pd
import graphviz

def plot_decision_boundaries(model, X, y, feature_names, target_names, feature_indices=(0, 1), resolution=0.02):
    """
    Plot decision boundaries of a model for 2 features.
    
    Parameters:
    -----------
    model : estimator with fit and predict methods
    X : feature dataset
    y : target vector
    feature_names : list of feature names
    target_names : list of class names
    feature_indices : tuple of two feature indices to plot (default: (0, 1))
    resolution : resolution of the mesh grid (default: 0.02)
    """
    # Extract the two features we will use
    X_reduced = X[:, feature_indices]
    
    # Create a mesh grid
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    # Create the full input array for prediction
    full_X = np.zeros((xx.ravel().shape[0], X.shape[1]))
    full_X[:, feature_indices[0]] = xx.ravel()
    full_X[:, feature_indices[1]] = yy.ravel()
    
    # Make predictions
    Z = model.predict(full_X)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='rainbow')
    
    # Plot the data points
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, 
                         s=20, edgecolor='k', cmap='rainbow')
    
    plt.xlabel(feature_names[feature_indices[0]])
    plt.ylabel(feature_names[feature_indices[1]])
    plt.title('Decision Boundary')
    plt.legend(handles=scatter.legend_elements()[0], labels=target_names)
    plt.tight_layout()
    
    return plt

def visualize_tree(model, feature_names, class_names):
    """
    Visualize a decision tree model.
    
    Parameters:
    -----------
    model : fitted decision tree model
    feature_names : list of feature names
    class_names : list of class names
    
    Returns:
    --------
    graphviz.Source : graphviz source code for the tree visualization
    """
    dot_data = export_graphviz(model, out_file=None, 
                              feature_names=feature_names,
                              class_names=class_names,
                              filled=True, rounded=True,
                              special_characters=True)
    return graphviz.Source(dot_data)

def plot_feature_importance(model, feature_names, title='Feature Importance'):
    """
    Plot feature importance from a tree-based model.
    
    Parameters:
    -----------
    model : fitted model with feature_importances_ attribute
    feature_names : list of feature names
    title : title for the plot (default: 'Feature Importance')
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    
    return plt

def plot_confusion_matrices(models, X_test, y_test, model_names, target_names):
    """
    Plot confusion matrices for multiple models.
    
    Parameters:
    -----------
    models : list of fitted models
    X_test : test feature set
    y_test : test target vector
    model_names : list of model names
    target_names : list of class names
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(n_models*6, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names,
                   ax=axes[i])
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        axes[i].set_title(f'Confusion Matrix - {name}')
    
    plt.tight_layout()
    return fig