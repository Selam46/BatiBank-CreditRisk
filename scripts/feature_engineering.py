import pandas as pd
from sklearn.preprocessing import LabelEncoder

def create_aggregate_features(data):
    """
    Create aggregate features for each customer.
    
    Parameters:
        data (DataFrame): The raw transaction dataset.
    
    Returns:
        DataFrame: A DataFrame containing aggregate features.
    """
    aggregate_features = data.groupby('CustomerId').agg(
        total_transaction_amount=('Amount', 'sum'),
        avg_transaction_amount=('Amount', 'mean'),
        transaction_count=('TransactionId', 'count'),
        std_transaction_amount=('Amount', 'std')
    ).reset_index()
    
    # Handle NaN values in standard deviation
    aggregate_features['std_transaction_amount'].fillna(0, inplace=True)
    
    return aggregate_features


def extract_date_features(data, datetime_column):
    """
    Extracts hour, day, month, and year from a datetime column.
    
    Parameters:
        data (DataFrame): The input dataset.
        datetime_column (str): The name of the datetime column.
    
    Returns:
        DataFrame: Dataset with new columns for hour, day, month, and year.
    """
    data[datetime_column] = pd.to_datetime(data[datetime_column], errors='coerce')
    data['transaction_hour'] = data[datetime_column].dt.hour
    data['transaction_day'] = data[datetime_column].dt.day
    data['transaction_month'] = data[datetime_column].dt.month
    data['transaction_year'] = data[datetime_column].dt.year
    return data

def encode_categorical(data, column, encoding_type='label'):
    """
    Encodes a categorical column using label encoding or one-hot encoding.
    
    Parameters:
        data (DataFrame): The input dataset.
        column (str): The name of the categorical column.
        encoding_type (str): Type of encoding ('label' or 'one-hot').
    
    Returns:
        DataFrame: Updated dataset with encoded columns.
    """
    if encoding_type == 'label':
        label_encoder = LabelEncoder()
        data[f'{column}_encoded'] = label_encoder.fit_transform(data[column])
        return data, dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    elif encoding_type == 'one-hot':
        dummies = pd.get_dummies(data[column], prefix=column)
        data = pd.concat([data, dummies], axis=1)
        return data
    else:
        raise ValueError("Invalid encoding type. Choose 'label' or 'one-hot'.")
