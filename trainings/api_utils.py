import os
from pathlib import Path


def generate_labels(dataset_path: str):
    dataset_path = Path(dataset_path) 
    if not dataset_path.exists():
        raise FileNotFoundError(f"Path '{dataset_path}' does not exist.")

    class_names = sorted([folder.name for folder in dataset_path.iterdir() if folder.is_dir()])
    
    labels = {str(i): class_name for i, class_name in enumerate(class_names)}

    return labels