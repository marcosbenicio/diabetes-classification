from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def grid_search_and_predict(model_instance, param_grid, X_train, y_train, X_val, y_val, scoring='accuracy', cv=10):
    """
    Perform hyperparameter tuning using GridSearchCV and make predictions with the best estimator.

    Parameters:
    - model_instance: instance of the machine learning model to tune.
    - param_grid (dict): Hyperparameters to tune.
    - X_train (array-like): Training data features.
    - y_train (array-like): Training data labels.
    - X_val (array-like): Validation data features.
    - y_val (array-like): Validation data labels.
    - scoring (str): Scoring metric to use for evaluation.
    - cv (int): Number of cross-validation folds.

    Returns:
    - best_params (dict): Best hyperparameters found during GridSearch.
    - best_score (float): Best score achieved with the best hyperparameters.
    - best_estimator (estimator): The best estimator from GridSearchCV.
    - y_pred (array-like): Predicted labels for the validation set.
    - val_accuracy (float): Accuracy of the best estimator on the validation set.
    """

    grid_search = GridSearchCV(model_instance, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # best hyperparameters, score, and estimator
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_estimator = grid_search.best_estimator_
    
    # Make predictions with the best estimator
    y_pred = best_estimator.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred)
    
    return best_params, best_score, best_estimator, y_pred, val_accuracy

def train_and_evaluate(model, params, X_train, y_train, X_eval, y_eval):
    """
    Train a machine learning model with given parameters, make predictions, and evaluate accuracy on the validation set.

    Parameters:
    - model: The machine learning model class to be used.
    - params (dict): Parameters to initialize the model.
    - X_train (array-like): Training data features.
    - y_train (array-like): Training data labels.
    - X_eval (array-like): Evaluation data features (can be validation or test set).
    - y_eval (array-like): Evaluation data labels (can be validation or test set).

    Returns:
    - eval_accuracy (float): Accuracy score of the model on the evaluation set.
    - y_pred_eval (array-like): Predictions made by the model on the evaluation set.
    """
    
    # Initialize the model with the given parameters
    model_instance = model(**params)
    
    # Train the model and predict on the validation set
    model_instance.fit(X_train, y_train)
    y_pred = model_instance.predict(X_eval)

    accuracy = accuracy_score(y_eval, y_pred)
    
    return accuracy, y_pred