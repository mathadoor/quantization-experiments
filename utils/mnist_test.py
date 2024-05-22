from mnist_shared import AverageMeter, SimpleCNN, download_mnist, val_mnist
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import mlflow

if __name__ == "__main__":
    # initialize mlflow
    uri = mlflow.get_tracking_uri()
    if uri is None:
        mlflow.set_tracking_uri('http://127.0.0.1:8080')
    client = mlflow.tracking.MlflowClient()

    # get model
    model_uri = client.get_model_version_download_uri("model_mnist", 1)
    model = mlflow.pytorch.load_model(model_uri)

    model.quantize()
    print(model.parameters())