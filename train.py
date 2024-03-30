import torch
from tqdm import tqdm
from src.utils.mlflow_utils import log_metrics, log_params, log_model
from src.data.data_preprocessing import ProjectDataPreprocessing
from src.models.model_architecture import LenetModel
from src.models.optimizer import Optimizer
import yaml

import mlflow

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def main(config, model, optimizer, loss_function, train_loader, test_loader):
    """
    Trains input model given dataloaders, optimizer and loss function,
    Tracking with MLFlow
    """
    with mlflow.start_run():
        mlflow.set_experiment("object-detection-project")
        log_params(config)
        best_loss = 1e+6

        for epoch in tqdm(range(config['train']['num_epochs']), desc = 'Model training..' ):
            loss = 0
            train_loss = 0
            for image, label in train_loader:
                image = image.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                model.train()
                outputs = model(image)
                loss = loss_function(outputs, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * image.size(0)
            train_loss=round(train_loss/len(train_loader.dataset), 5)

            loss = 0
            val_loss = 0
            with torch.inference_mode():
                for image, label in val_loader:
                    image = image.to(device)
                    label = label.to(device)
                    model.eval()
                    outputs = model(image)
                    loss = loss_function(outputs, label)
                    val_loss += loss.item() * image.size(0)
                val_loss=round(val_loss/len(val_loader.dataset), 5)
            print(f"Train loss: {train_loss} | Test loss: {val_loss}")
            log_metrics({"train_loss": train_loss,
                         "val_loss": val_loss})
            print('eiojf')
            log_model(model, "model")
            print('jdfnkls')
            return model

if __name__ == "__main__":
    data_processor = ProjectDataPreprocessing()
    train_loader, test_loader, val_loader = data_processor.preprocess_data(config)
    model = LenetModel(config)
    optimizer = Optimizer(config).optim(model)
    loss_function = torch.nn.SmoothL1Loss()
    main(config, model, optimizer, loss_function, train_loader, test_loader)