import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler



def convert_dtypes(df: pd.DataFrame, dtype_dict: dict) -> pd.DataFrame:
    """
    Convert the data types of specified columns.

    Parameters:
        df (pd.DataFrame): The original DataFrame.
        dtype_dict (dict): Dictionary specifying the columns and their corresponding data types.
                           Key: Desired data type (e.g., 'bool', 'int32', 'float32')
                           Value: List of columns to be converted to the key data type

    Returns:
        pd.DataFrame: DataFrame with new data types.

    Example:
        dtype_dict = {
            'float32': ['ColumnA', 'ColumnB'],
            'int32': ['ColumnC'],
            'bool': ['ColumnD', 'ColumnE']
        }
        df = convert_dtypes(df, dtype_dict)
    """
    for dtype, columns in dtype_dict.items():
        for column in columns:
            df[column] = df[column].astype(dtype)
    return df

def categorical_encoding(df_train , df_val , df_test ,features = None , sparse = False ):
    """
    Function to preprocess data frames based on the selected features.
    
    Parameters:
    - df_train (pd.DataFrame): The training dataframe. If None, no encoding is performed for training.
    - df_val (pd.DataFrame): The validation dataframe. If None, no encoding is performed for validation.
    - df_test (pd.DataFrame): The test dataframe. If None, no encoding is performed for testing.
    - features (list, optional): List of features to be selected from the data frames. If None, all columns are used.
    - sparse (bool): Whether to use sparse matrices or not.

    Returns:
    - X_train_encoded (np.ndarray): The encoded features for the training data.
    - X_val_encoded (np.ndarray): The encoded features for the validation data.
    - X_test_encoded (np.ndarray): The encoded features for the test data.
    - dv (DictVectorizer): The fitted DictVectorizer instance used for encoding.
    """
    
    # If features are not provided, use all columns from df_train
    if features is None and df_train is not None:
        features = df_train.columns.tolist()

    # Create copies of the dataframes with only the selected features
    dfs = {'train': df_train, 'val': df_val, 'test': df_test}
    dfs = {key: df[features].copy() if df is not None else None for key, df in dfs.items()}

    dv = DictVectorizer(sparse=sparse)
    transformed_data = {}
    fitted = False

    for key, df in dfs.items():
        if df is not None:
            df_dict = df.to_dict(orient='records')
            if not fitted:
                transformed_data[key] = dv.fit_transform(df_dict)
                fitted = True
            else:
                transformed_data[key] = dv.transform(df_dict)

    return (transformed_data.get('train'), transformed_data.get('val'),
            transformed_data.get('test'), dv)

def preprocess_and_encode_features(df_train, df_val, df_test, selected_features: list,  
                                   feature_to_dtypes: dict = None, features_to_scale: list = None):
    """
    Preprocess the training, validation, and test dataframes by standardizing numerical features and encoding categorical variables.

    This function scales selected numerical features and encodes categorical variables using one-hot encoding.
      It is designed to be used after feature selection during exploratory data analysis (EDA).

    Parameters:
    - df_train (pd.DataFrame): The training dataframe.
    - df_val (pd.DataFrame): The validation dataframe.
    - df_test (pd.DataFrame): The test dataframe.
    - selected_features (list): List of features selected for modeling.
    - feature_to_dtypes (dict, optional): Dictionary mapping features to their desired data types for conversion.
    - features_to_scale (list, optional): List of numerical features that need to be standardized.

    Returns:
    - X_train_encoded (np.ndarray): The encoded features for the training data.
    - X_val_encoded (np.ndarray): The encoded features for the validation data.
    - X_test_encoded (np.ndarray): The encoded features for the test data.
    - encoder (DictVectorizer): The fitted DictVectorizer instance used for encoding.
    """
    
    dfs = [df_train, df_val, df_test]
    
    # Standardize features_to_scale if provided
    if features_to_scale:
        scaler = StandardScaler()
        scaler.fit(df_train[features_to_scale])
        for df in dfs:
            df[features_to_scale] = scaler.transform(df[features_to_scale])
    
    # Convert data types if feature_dtypes is provided
    if feature_to_dtypes:
        for df in dfs:
            df = convert_dtypes(df, feature_to_dtypes)

    # Encode categorical variables for selected features from EDA
    X_train_encoded, X_val_encoded, X_test_encoded, encoder = categorical_encoding(df_train, df_val, df_test, selected_features)

    return X_train_encoded, X_val_encoded, X_test_encoded, encoder


def feature_elimination(df_train, df_val, Y_train, Y_val, model, metric_func, features, base_metric):
    """
    Function to perform feature elimination based on the given model and metric function.

    Parameters:
    - df_train, df_val: pandas.DataFrame
        Training and validation datasets.
    - Y_train, Y_val: pandas.Series or numpy.ndarray
        Target values for training and validation datasets.
    - model: scikit-learn estimator
        The machine learning model to be trained.
    - metric_func: callable
        The metric function to evaluate model performance.
    - base_metric: float
        The base accuracy or metric value to compare against.

    Returns:
    - df_feature_metrics: pandas.DataFrame
        DataFrame showing the impact of each feature's elimination on model performance.
    """
    
    eliminated_features = []
    metric_list = []
    metric_diff_list = []   # to store the difference
    metric_name = metric_func.__name__

    for feature in features:

        df_train_drop = df_train.drop(feature, axis=1, inplace=False)
        df_val_drop = df_val.drop(feature, axis=1, inplace=False)
        X_train_small, X_val_small, _, _ = categorical_encoding(df_train_drop, df_val_drop, df_val_drop)

        model.fit(X_train_small, Y_train)
        Y_pred = model.predict(X_val_small)
        metric_score = metric_func(Y_pred, Y_val)
        
        # Store results in lists
        eliminated_features.append(feature)
        metric_list.append(metric_score)
        metric_diff_list.append(abs(base_metric - metric_score))  # compute the difference

    df_feature_metrics = pd.DataFrame({ 
        'eliminated_feature': eliminated_features,
        f'{metric_name}': metric_list,
        f'{metric_name}_diff': metric_diff_list  # add the difference as a column
    })\
    .sort_values(by=[f'{metric_name}_diff'], ascending=False)

    return df_feature_metrics
