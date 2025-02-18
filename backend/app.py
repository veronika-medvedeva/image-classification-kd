import io
import os
import json
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, HTTPException, File, UploadFile
from PIL import Image
from trainings.api_utils import generate_labels
from trainings.student import SimpleNet
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil

device = torch.device("cpu")

LABELS_PATH = "dataset/fruits-360/Training"
labels = generate_labels(LABELS_PATH)

transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

MODEL_PATH = "models/student_model.pth"

def load_model(path: str, device):
    model = SimpleNet().to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

model = load_model(MODEL_PATH, device)

app = FastAPI(title="FastAPI Image Classification")

origins = [
    "http://localhost:8080",  
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "FastAPI image model server running."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    with open(f"./uploads/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    image = Image.open(f"./uploads/{file.filename}").convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image) 
        predicted_class_idx = torch.argmax(outputs, dim=1).item()
    
    predicted_label = labels.get(str(predicted_class_idx), "Unknown") 
    confidence = torch.exp(outputs[0, predicted_class_idx]).item()
    
    return JSONResponse(content={
        "predicted_class": predicted_label,
        "confidence": confidence  
    })
