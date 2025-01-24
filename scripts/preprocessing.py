import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

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
 
