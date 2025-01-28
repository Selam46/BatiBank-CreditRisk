import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_data(file_path):
    return pd.read_csv(file_path)

def overview(data):
    print(f"Shape: {data.shape}")
    print(data.info())
    print(f"Duplicate rows: {data.duplicated().sum()}")

def plot_missing(data):
    msno.matrix(data)
    plt.show()

    msno.heatmap(data)
    plt.show()

def plot_numerical_distributions(data):
    numerical_cols = data.select_dtypes(include=np.number)
    numerical_cols.hist(figsize=(12, 10), bins=20, color='skyblue', edgecolor='black')
    plt.tight_layout()
    plt.show()

def plot_categorical_distributions(data):
    categorical_cols = data.select_dtypes(include='object')
    for col in categorical_cols.columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(y=data[col], order=data[col].value_counts().index, palette="Blues_r")
        plt.show()




def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handles missing values in the dataset using specified strategy.
    
    Parameters:
        data (DataFrame): The input dataset.
        strategy (str): The imputation strategy ('mean', 'median', 'mode', or 'remove').
        columns (list): Columns to apply the strategy to.
    
    Returns:
        DataFrame: Dataset with missing values handled.
    """
    if strategy == 'mean':
        for col in columns:
            data[col] = data[col].fillna(data[col].mean())
    elif strategy == 'median':
        for col in columns:
            data[col] = data[col].fillna(data[col].median())
    elif strategy == 'mode':
        for col in columns:
            data[col] = data[col].fillna(data[col].mode()[0])
    elif strategy == 'remove':
        data = data.dropna()
    else:
        raise ValueError("Invalid strategy. Choose 'mean', 'median', 'mode', or 'remove'.")
    return data

def normalize_features(data, columns):
    """
    Normalizes numerical features to the range [0, 1].
    
    Parameters:
        data (DataFrame): The input dataset.
        columns (list): Columns to normalize.
    
    Returns:
        DataFrame: Dataset with normalized features.
    """
    scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

def standardize_features(data, columns):
    """
    Standardizes numerical features to have mean 0 and standard deviation 1.
    
    Parameters:
        data (DataFrame): The input dataset.
        columns (list): Columns to standardize.
    
    Returns:
        DataFrame: Dataset with standardized features.
    """
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

 
