import torch
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

def count_params(model):
    return sum((p != 0).sum().item() for p in model.parameters() if p.requires_grad)

def calc_pytorch_weights(model):
    result = 0
    for layer in model.modules():
        if hasattr(layer, 'weight_mask'):
            result += int(layer.weight_mask.sum().item())
        elif hasattr(layer, 'weight'):
            result += layer.weight.numel()
    return result

def register_check_model(mode="teacher"):
    
    client = MlflowClient()
    experiment_name = f"experiment-{mode}"
    reg_model_name = f"{mode}-model"
    
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found.")
        return
    
    try:
        client.create_registered_model(reg_model_name)
        print(f"Registered new model: {reg_model_name}")
    except Exception as e:
        print(f"Model already exists or error occurred: {e}")
    
    runs = client.search_runs([experiment.experiment_id], run_view_type=ViewType.ACTIVE_ONLY)
    if not runs:
        print("No runs found in experiment.")
        return
    
    run_id = runs[0].info.run_id
    artifact_uri = runs[0].info.artifact_uri
    current_f1_score = runs[0].data.metrics.get("f1_score", 0)
    
    existing_models = client.search_model_versions(f"name='{reg_model_name}'")
    prod_model = next((m for m in existing_models if m.current_stage == "Production"), None)
    
    if not prod_model or current_f1_score > mlflow.get_run(prod_model.run_id).data.metrics.get("f1_score", 0):
        result = client.create_model_version(
            name=reg_model_name,
            source=f"{artifact_uri}/model",
            run_id=run_id
        )
        client.transition_model_version_stage(
            name=reg_model_name,
            version=result.version,
            stage="Production"
        )
        print(f"Updated Production model to version {result.version}")
    else:
        print("Existing Production model is better, no update needed.")
        return

    latest_version = client.get_latest_versions(reg_model_name, stages=["Production"])[0].version
    model = mlflow.pytorch.load_model(f"models:/{reg_model_name}/{latest_version}")

    save_path = f"models/{mode}_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Production model saved to '{save_path}'.")
