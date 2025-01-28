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


import category_encoders as ce

def compute_rfms(data, transaction_col, amount_col, fraud_col, customer_col):
    """
    Compute RFMS (Recency, Frequency, Monetary, Severity) features for customers.
    
    Parameters:
        data (DataFrame): Input dataset.
        transaction_col (str): Column name for transaction datetime.
        amount_col (str): Column name for transaction amounts.
        fraud_col (str): Column name for fraud results.
        customer_col (str): Column name for customer IDs.
    
    Returns:
        DataFrame: RFMS features for each customer.
    """
    # Ensure the transaction column is in datetime format
    data[transaction_col] = pd.to_datetime(data[transaction_col])
    
    # Aggregate data to calculate RFMS features
    rfms = data.groupby(customer_col).agg(
        recency=(transaction_col, lambda x: (data[transaction_col].max() - x.max()).days),
        frequency=(customer_col, 'count'),
        monetary=(amount_col, 'sum'),
        severity=(fraud_col, 'mean')
    ).reset_index()
    
    # Replace NaN values in 'severity' with 0
    rfms['severity'] = rfms['severity'].fillna(0)
    
    return rfms

def apply_woe_binning(data, features, label_col):
    """
    Perform WoE binning on selected features.
    
    Parameters:
        data (DataFrame): Input dataset.
        features (list): List of feature columns for WoE transformation.
        label_col (str): Column name for the target variable (Good/Bad label).
    
    Returns:
        DataFrame: WoE-transformed dataset.
    """
    # Initialize WoE encoder from the category_encoders package
    woe_encoder = ce.WOEEncoder(cols=features)
    
    # Fit the WoE encoder and transform the data
    data_woe = woe_encoder.fit_transform(data[features], data[label_col])
    
    # Return the WoE-transformed features alongside the original data
    data_woe = pd.concat([data.drop(columns=features), data_woe], axis=1)
    
    return data_woe
