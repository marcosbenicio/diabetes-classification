
import pandas as pd
from sklearn.metrics import precision_score,recall_score ,roc_auc_score, f1_score

def classification_metrics(y_true, y_pred, column_name = 'Value'):

    """
    Compute classification metrics and return them as a DataFrame.

    This function calculates precision, recall, AUC (Area Under Curve), and F1 score for the given true and predicted labels.
    
    Parameters:
    - y_true (array-like): True labels of the data.
    - y_pred (array-like): Predicted labels by the model.
    - column_name (str, optional): Name to be used for the metrics column in the returned DataFrame. Default is 'Value'.
    
    Returns:
    - metrics_df (pd.DataFrame): DataFrame with a metrics index and a single column with the computed metrics. 
                                 The index contains the names of the metrics, and the column contains the corresponding scores.
    """
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Create DataFrame
    metrics_df = pd.DataFrame({
        'metrics': ['Precision', 'Recall', 'AUC', 'F1 Score'],
        column_name: [precision, recall, auc, f1]
    }).set_index('metrics')
    
    return metrics_df