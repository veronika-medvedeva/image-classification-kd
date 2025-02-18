import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.entities import ViewType
from sklearn.metrics import accuracy_score, f1_score
from trainings.teacher import TeacherModel
from trainings.training_utils import train_teacher_model
from trainings.utils import register_check_model
from dataset.dataloader import get_dataloaders

def main(args):
    """Train the teacher model and log to MLflow."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataloaders = get_dataloaders(args.batch_size, args.data_dir)
    model = TeacherModel(feature_extract=args.feature_extract).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    mlflow.set_experiment("experiment-teacher")

    with mlflow.start_run() as run:
        mlflow.log_params({
            "learning_rate": args.lr,
            "momentum": args.momentum,
            "batch_size": args.batch_size,
            "num_epochs": args.epochs,
            "feature_extract": args.feature_extract
        })

        model, accuracy, f1 = train_teacher_model(model, dataloaders, criterion, optimizer, num_epochs=args.epochs, device=device)

        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})
        mlflow.pytorch.log_model(model, "model")

    client = MlflowClient()
    
    register_check_model(mode="teacher")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teacher model training")
    parser.add_argument("--data_dir", type=str, default="dataset/fruits-360")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--feature_extract", type=bool, default=True, help="Freeze layers except final layer")

    args = parser.parse_args()
    main(args)