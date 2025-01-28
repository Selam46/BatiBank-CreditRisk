import joblib

def load_model(path):
    """
    Load a saved machine learning model.
    
    Parameters:
        path (str): Path to the model file.
    
    Returns:
        Model: Loaded machine learning model.
    """
    return joblib.load(path)
