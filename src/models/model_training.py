import torch
from tqdm import tqdm
from utils.mlflow_utils import log_metrics, log_params, log_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_loop(model, train_loader, val_loader, optimizer, loss_function):
    """
    Trains input model given dataloaders, optimizer and loss function,
    Tracking with MLFlow
    """
    with mlflow.start_run():
        mlflow.set_experiment("object-detection-project")
        log_params(self.params)

        for epoch in tqdm(range(100), desc = 'Model training..' ):
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
                train_loss += loss * inputs.size(0)
            train_loss=train_loss/len(train_loader.dataset)

            loss = 0
            val_loss = 0
            with torch.inference_mode():
                for image, label in val_loader:
                    image = image.to(device)
                    label = label.to(device)
                    model.eval()
                    outputs = model(image)
                    loss = loss_function(outputs, label)
                    val_loss += loss * inputs.size(0)
                val_loss=val_loss/len(val_loader.dataset)
                if val_loss < best_loss:
                    print('Saving best model')
                    torch.save(model.state_dict(), '../../models/{}.pt'.format(model_name))
                    best_loss = val_loss
            log_metrics({"train_loss": train_loss,
                         "val_loss": val_loss})
            log_model(model, "model")
            return model