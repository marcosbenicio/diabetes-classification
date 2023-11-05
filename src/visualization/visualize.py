import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_feature_importance(df, x, y, ax, threshold=0.002, pad=5.0, title='Feature Importance', 
                            xlabel='Features', ylabel='Importance', palette=None):
    """
    Function to plot the feature importance with a distinction of importance based on a threshold.

    Parameters:
    - df: pandas.DataFrame
        DataFrame containing features and their importance scores.
    - x: str
        Name of the column representing feature names.
    - y: str
        Name of the column representing feature importance scores.
    - ax: matplotlib axis object
        Axis on which to draw the plot.
    - threshold: float, optional (default=0.002)
        Value above which bars will be colored differently.
    - pad: float, optional (default=5.0)
        Adjust the layout of the plot.
    - title: str, optional (default='Feature Importance')
        Title of the plot.
    - xlabel: str, optional (default='Features')
        Label for the x-axis.
    - ylabel: str, optional (default='Importance')
        Label for the y-axis.
    - palette: list, optional
        A list of two colors. The first color is for bars below the threshold and the second is for bars above.

    Returns:
    - None (modifies ax in-place)
    """
    if palette is None:
        palette = ["blue", "red"]
    
    blue, red = palette
    colors = [red if value >= threshold else blue for value in df[y]]
    sns.barplot(x=x, y=y, data=df, ax=ax, alpha=0.5, palette=colors)
    ax.set_xticklabels(df[x], rotation=70, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=15) 
    ax.grid(axis='y')
    ax.set_title(title, fontsize=15)
    plt.tight_layout(pad=pad)


def plot_confusion_matrix(Y_true, Y_pred, title, ax, xy_legends):
    """
    Plot the confusion matrix for given true and predicted labels.

    Parameters:
    - Y_true: Actual labels
    - Y_pred: Predicted labels
    - title: Title for the plot
    - ax: Axis object to plot on
    """
    cm = confusion_matrix(Y_true, Y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='coolwarm',
                xticklabels= xy_legends, yticklabels= xy_legends,
                alpha=0.5, cbar=False, ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')