import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from fastapi import FastAPI
from pydantic import BaseModel
import random

# Load product data
inventory = pd.read_excel("Data/Pieces.xlsx")
purchases = pd.read_excel("Data/Purchases.xlsx")

# Encode categorical features
label_encoders = {}
for col in ["Room Type", "Aesthetic", "Category", "Color"]:
    le = LabelEncoder()
    inventory[col] = le.fit_transform(inventory[col])
    label_encoders[col] = le

# Generate embeddings using Hugging Face
model = SentenceTransformer("all-MiniLM-L6-v2")
inventory["embedding"] = inventory["Description"].apply(lambda x: model.encode(x))

# Define recommendation model
class RecommendationNN(nn.Module):
    def __init__(self, input_size):
        super(RecommendationNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Convert data into training format
X_train = np.array(list(inventory["embedding"]))
y_train = np.random.rand(len(X_train))  # Placeholder, should be based on user interactions

# Train model
model_nn = RecommendationNN(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model_nn.parameters(), lr=0.001)

def train_model():
    for _ in range(10):  # 10 epochs for now
        optimizer.zero_grad()
        outputs = model_nn(torch.tensor(X_train, dtype=torch.float32))
        loss = criterion(outputs.squeeze(), torch.tensor(y_train, dtype=torch.float32))
        loss.backward()
        optimizer.step()
    print("Training Complete")

train_model()

# FastAPI deployment
app = FastAPI()
class UserRequest(BaseModel):
    user_id: int

def recommend_products(user_id):
    user_purchases = purchases[purchases["User ID"] == user_id]["ID"]
    recommendations = []
    for _, item in inventory.iterrows():
        item_vector = np.array(item["embedding"])
        score = model_nn(torch.tensor(item_vector, dtype=torch.float32)).item()
        recommendations.append((item["ID"], score))
    
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    return recommendations[:10]

@app.post("/recommend/")
def get_recommendations(request: UserRequest):
    return {"recommendations": recommend_products(request.user_id)}

# Run with: uvicorn script_name:app --reload
