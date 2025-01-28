from sklearn.metrics import classification_report, roc_auc_score

def train_model(model, X_train, y_train, X_test, y_test):
    """
    Train and evaluate a model.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    return model, report, roc_auc
