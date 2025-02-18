import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(batch_size, data_dir, input_size=299, val_split=0.2):
    """Loads datasets and returns train, validation, and test DataLoaders."""

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/Training", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/Test", transform=transform)

    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    dataloaders ={'train': train_loader,'val': val_loader, 'test': test_loader}

    return dataloaders
