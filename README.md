# Image Classification with Knowledge Distillation (PyTorch)

## Overview
This project utilizes **knowledge distillation** to train a smaller, efficient neural network for **image classification** using the **Fruits-360 dataset**. The goal is to transfer knowledge from a larger, pre-trained **teacher model** to a smaller **student model**, making it suitable for deployment on less powerful hardware.

---

## Dataset: Fruits-360
The **Fruits-360 dataset** contains images of various **fruits and vegetables**. The dataset is not included in this repository; you can download it from **[Kaggle](https://www.kaggle.com/datasets/moltean/fruits/)**.

### Dataset Properties:
- **Total Images:** 94,110
- **Training Set:** 70,491 images (one fruit or vegetable per image)
- **Test Set:** 23,619 images (one fruit or vegetable per image)
- **Number of Classes:** 141 (fruits, vegetables, and nuts)
- **Image Size:** 100x100 pixels

### Example Classes:
- **Apples:** Crimson Snow, Golden, Granny Smith, Pink Lady, Red Delicious
- **Bananas:** Yellow, Red, Lady Finger
- **Berries:** Blueberry, Raspberry, Redcurrant, Mulberry
- **Citrus:** Orange, Lemon, Lime, Grapefruit, Clementine, Tangelo
- **Exotic Fruits:** Avocado, Durian, Lychee, Mangostan, Pitahaya Red
- **Vegetables:** Cauliflower, Beetroot, Corn, Eggplant, Onion, Potato

---

## Knowledge Distillation
**Knowledge distillation** transfers knowledge from a large **teacher model** to a smaller **student model**. The student model benefits from distilled knowledge, achieving competitive performance with reduced computational requirements.

---

## Project Structure
This project leverages **Docker Compose** to orchestrate multiple services, ensuring seamless integration and efficient management.

### Key Components:
- **MLflow Server** – Manages experiment tracking and artifact storage.
- **Teacher Model Training (InceptionV3)** – Trains a high-performance teacher model for knowledge distillation.
- **Student Model Training (Simple NN)** – Trains a lightweight student model using **knowledge distillation** and **pruning techniques**.
- **Backend API (FastAPI)** – Exposes trained models via a **REST API** for inference.
- **Frontend (Web Client)** – Provides a user-friendly interface to interact with the backend and make predictions.

---

## Getting Started

### 1. Start the MLflow Server
```sh
docker-compose up --build -d mlflow
```
- **Access MLflow server:** [http://localhost:5001](http://localhost:5001)

### 2. Train the Teacher Model
The **teacher model** is based on **InceptionV3**, optimized using **SGD**. The best model is automatically replaced in **MLflow Production**.

#### Command:
```sh
docker-compose run --rm trainteacher
```
#### Additional Arguments:
- `data_dir` (default: `dataset/fruits-360`)
- `batch_size` (default: `16`)
- `epochs` (default: `1`)
- `lr` (learning rate, default: `0.02`)
- `momentum` (default: `0.9`)
- `feature_extract` (True to train only final layer, False for full model training)

#### Example Usage:
```sh
docker-compose run --rm trainteacher --epochs 5 --lr 0.01
```
- The best-trained **teacher model** is saved in the `models` folder (`teacher_model.pth`).

### 3. Train the Student Model
The **student model** is a simple neural network, optionally **pruned** for efficiency. It is trained using the **Adam optimizer** and **knowledge distillation techniques**.

#### Command:
```sh
docker-compose run --rm trainstudent
```
#### Additional Arguments:
- `data_dir` (default: `dataset/fruits-360`)
- `batch_size` (default: `16`)
- `epochs` (default: `1`)
- `lr` (default: `0.001`)
- `alpha` (default: `0.5`)
- `temperature` (default: `2.0`)
- `prune_rate` (default: `0.9`)
- `prune_method` (default: `'l1_unstructured'` | `'global'` | `'random'` | `'structured'`)

#### Example Usage:
```sh
docker-compose run --rm trainstudent --epochs 5 --lr 0.01
```
> **Note:** Train the student model only **after training the teacher model**.
- The best-trained **student model** is saved in the `models` folder (`student_model.pth`).

---

## Running the Prediction Server
Start the **frontend and backend servers**:
```sh
docker-compose up --build frontend
```
- **Backend will start automatically**.
- **Check server status:** [http://localhost:5050](http://localhost:5050)

### Making Predictions
1. Visit: [http://localhost:8080](http://localhost:8080)
2. Upload an image
3. Click **Upload & Predict**

---

## Contributing
Feel free to open issues and submit pull requests. Any contributions are welcome!

---

## License
This project is licensed under the **MIT License**.

---

## Acknowledgments
Special thanks to:
- [Fruits-360 Dataset](https://www.kaggle.com/datasets/moltean/fruits/)
- The PyTorch and FastAPI communities
- Docker and MLflow for model tracking and deployment

