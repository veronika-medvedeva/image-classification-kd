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
from dataset.dataloader import get_dataloaders
from trainings.student import SimpleNet
from trainings.utils import count_params, calc_pytorch_weights, register_check_model
from trainings.training_utils import train_student_model

def main(args):
    """Train the student model and log to MLflow."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataloaders = get_dataloaders(args.batch_size, args.data_dir)

    model_t = TeacherModel().to(device)
    checkpoint = torch.load('models/teacher_model.pth', map_location=device)
    model_t.load_state_dict(checkpoint)
    model_t.eval()

    model_s = SimpleNet()

    n_params = count_params(model_s)
    weights = calc_pytorch_weights(model_s)
    print('{} parameters in student model'.format(n_params))
    print('{} weights in student model'.format(weights))

    model_s.prune(method=args.prune_method, rate=args.prune_rate)
    n_params = count_params(model_s)
    weights = calc_pytorch_weights(model_s)

    print('After pruning there are {} parameters in student model'.format(n_params))
    print('After pruning there are {} weights in student model'.format(weights))

    model_s.to(device)

    optimizer = torch.optim.Adam(model_s.parameters(), lr=args.lr)
    
    mlflow.set_experiment("experiment-student")

    with mlflow.start_run() as run:
        mlflow.log_params({
            "learning_rate": args.lr,
            "alpha": args.alpha,
            "temperature" : args.temperature,
            "batch_size": args.batch_size,
            "num_epochs": args.epochs,
            "prune_rate": args.prune_rate,
            "prune_method": args.prune_method
        })

        model_st, accuracy, f1 = train_student_model(model_s, dataloaders, optimizer, model_t, alpha=args.alpha, temperature=args.temperature, num_epochs=args.epochs, device=device)

        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})
        mlflow.pytorch.log_model(model_st, "model")

    client = MlflowClient()

    register_check_model(mode="student")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student model training")
    parser.add_argument("--data_dir", type=str, default="dataset/fruits-360")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--prune_rate", type=float, default=0.9)
    parser.add_argument("--prune_method", type=str, default="l1_unstructured", choices=["l1_unstructured", "global", "random", "structured"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)

    args = parser.parse_args()
    main(args)