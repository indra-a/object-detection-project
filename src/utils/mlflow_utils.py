import mlflow
import mlflow.pytorch

def log_model(model, artifact_path, metadata=None):
    """Log a PyTorch model to MLflow with optional metadata"""
    mlflow.pytorch.log_model(model, artifact_path, metadata=metadata)

def load_model(model_uri):
    """Load a PyTorch model from MLflow"""
    return mlflow.pytorch.load_model(model_uri)

def log_params(params):
    """Log parameters to MLflow"""
    mlflow.log_params(params)

def log_metrics(metrics):
    """Log metrics to MLflow"""
    mlflow.log_metrics(metrics)