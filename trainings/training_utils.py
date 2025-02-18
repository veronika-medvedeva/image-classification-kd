import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Teacher Model training functions
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), correct / total

def evaluate(model, loader, criterion, device, mode="Validation"):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=f"{mode}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if mode == "Test":
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

    if mode == "Test":
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="weighted")

    return total_loss / len(loader), correct / total

def train_teacher_model(model, dataloaders, criterion, optimizer, num_epochs, device):

    best_val_loss, best_model_wts = float("inf"), None

    for epoch in range(1, num_epochs + 1):
        logging.info(f"Epoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, dataloaders['val'], criterion, device, mode="Validation")

        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            logging.info("Best model updated.")
            best_val_loss = val_loss
            best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)

    test_acc, test_f1 = evaluate(model, dataloaders['test'], criterion, device, mode="Test")
    logging.info(f"Test Accuracy: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}")

    return model, test_acc, test_f1

# Student Model training functions
def compute_distillation_loss(inputs, labels, teacher, student, alpha, temperature):
    kld_loss = nn.KLDivLoss(reduction="batchmean")
    ce_loss = nn.CrossEntropyLoss()

    teacher_logits = teacher(inputs)
    student_logits = student(inputs)

    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_outputs = student_logits / temperature

    distillation_loss = kld_loss(soft_outputs, soft_targets)
    student_loss = ce_loss(student_logits, labels)

    return alpha * distillation_loss + (1 - alpha) * student_loss, student_logits

def train_student_one_epoch(model, train_loader, optimizer, device, teacher, alpha, temperature):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        loss, outputs = compute_distillation_loss(inputs, labels, teacher, model, alpha, temperature)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
        
    return total_loss / len(train_loader), correct / total

def evaluate_student_one_epoch(model, loader, device, teacher, alpha, temperature):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            loss, outputs = compute_distillation_loss(inputs, labels, teacher, model, alpha, temperature)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total

def evaluate_student(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Test"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="weighted")

def train_student_model(model, dataloaders, optimizer, teacher, alpha, temperature, num_epochs, device):
    best_val_loss, best_model_wts = float("inf"), None

    for epoch in range(1, num_epochs + 1):
        logging.info(f"Epoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_student_one_epoch(model, dataloaders['train'], optimizer, device, teacher, alpha, temperature)
        val_loss, val_acc = evaluate_student_one_epoch(model, dataloaders['val'], device, teacher, alpha, temperature)

        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            logging.info("Best model updated")
            best_val_loss = val_loss
            best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)

    test_acc, test_f1 = evaluate_student(model, dataloaders['test'], device)
    logging.info(f"Test Accuracy: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}")

    return model, test_acc, test_f1