# Plot Functions 

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels):
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize = (6, 4))
    sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, fmt = 'd')
    plt.title('Confusion matrix')
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.show()

    # Show the plot
    plt.show()



def plot_roc_auc(model_results, features, actual_labels):
    """
    A function to plot the ROC-AUC curve for binary classification models.

    Parameters:
    - model_results (list): List of tuples containing model names and corresponding fitted models.
    - features (array-like): Input features for prediction.
    - actual_labels (array-like): True labels for the binary classification.

    Returns:
    - None: Displays the ROC-AUC curve plot.
    """
    # Create a new figure and axis for the ROC-AUC plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Iterate over model results (name, fitted model)
    for model_name, model in model_results:
        # Predict probabilities using the model
        y_score = model.predict_proba(features)[:, 1]

        # Compute ROC curve values
        fpr, tpr, _ = metrics.roc_curve(actual_labels, y_score)

        # Calculate the area under the ROC curve
        roc_auc = metrics.auc(fpr, tpr)

        # Plot the ROC curve for the current model
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (area = %0.2f)" % roc_auc)

    # Plot the diagonal reference line for a random classifier
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Set plot limits and labels
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

    # Add a legend in the lower right corner
    plt.legend(loc="lower right")

    # Display the plot
    plt.show()

    
    
    
def scatter_custom(dataset, columns_list, rows, cols, suptitle):
    """
    Create scatter plots for numeric columns in a dataset.

    Parameters:
    - dataset: DataFrame, the dataset containing the columns to be plotted.
    - columns_list: list, the list of column names to be plotted.
    - rows: int, the number of rows in the subplot grid.
    - cols: int, the number of columns in the subplot grid.
    - suptitle: str, the title of the entire subplot.

    Returns:
    - None (displays the scatter plots).
    """
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    fig.suptitle(suptitle, fontsize=16)

    for i in range(rows):
        for j in range(cols):
            col_index = i * cols + j
            if col_index < len(columns_list):
                sns.scatterplot(x=dataset[columns_list[col_index]], y=dataset[columns_list[col_index]], ax=axes[i, j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust subplot layout
    plt.show()
    
    
# checking boxplots
def boxplots_custom(dataset, columns_list, rows, cols, suptitle):
    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(16,25))
    fig.suptitle(suptitle,y=1, size=25)
    axs = axs.flatten()
    for i, data in enumerate(columns_list):
        sns.boxplot(data=dataset[data], orient='h', ax=axs[i])
        axs[i].set_title(data + ', skewness is: '+str(round(dataset[data].skew(axis = 0, skipna = True),2)))
        

        
