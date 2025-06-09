import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler

# Load data once
inventory = pd.read_excel("Data/Pieces.xlsx", sheet_name="Pieces")

# Normalize price
scaler = MinMaxScaler()
inventory["Price"] = scaler.fit_transform(inventory[["Price"]])

# Color compatibility dictionary
color_matching = {
    "Black": ["Black", "White", "Gray", "Mahogany", "Walnut", "Beige", "Clear", "Natural", "Oak"],
    "White": ["White", "Beige", "Gray", "Black", "Clear", "Oak", "Natural", "Walnut"],
    "Gray": ["Gray", "Black", "White", "Beige", "Blue", "Mahogany", "Walnut", "Oak"],
    "Brown": ["Brown", "Beige", "Tan", "Walnut", "Mahogany", "Natural", "Oak", "Cherry"],
    "Blue": ["Blue", "Light Blue", "Gray", "White", "Beige", "Teal", "Black"],
    "Mahogany": ["Mahogany", "Brown", "Walnut", "Beige", "Cherry", "Black", "Natural"],
    "Beige": ["Beige", "White", "Brown", "Gray", "Walnut", "Natural", "Oak", "Clear"],
    "Clear": ["Clear", "White", "Gray", "Black", "Beige", "Natural", "Oak"],
    "Oak": ["Oak", "Walnut", "Natural", "Beige", "White", "Gray", "Mahogany", "Black"],
    "Walnut": ["Walnut", "Brown", "Mahogany", "Beige", "Oak", "Natural", "Black"],
    "Cherry": ["Cherry", "Mahogany", "Brown", "Beige", "Walnut", "Black"],
    "Teal": ["Teal", "Blue", "Gray", "White", "Beige", "Black"],
    "Natural": ["Natural", "Oak", "Walnut", "Beige", "White", "Clear", "Brown"]
}

# Attribute weights
weights = {
    "Room Type": 0.4,
    "Aesthetic": 0.3,
    "Category": 0.05,
    "Price": 0.1,
    "Color": 0.15
}

# Encoding dictionaries
room_map = {v: i for i, v in enumerate(inventory["Room Type"].unique(), 1)}
aesthetic_map = {v: i for i, v in enumerate(inventory["Aesthetic"].unique(), 1)}
category_map = {v: i for i, v in enumerate(inventory["Category"].unique(), 1)}

# Vectorize a furniture piece
def vectorize(row):
    return np.array([
        room_map.get(row["Room Type"], 0) / max(room_map.values()),
        aesthetic_map.get(row["Aesthetic"], 0) / max(aesthetic_map.values()),
        category_map.get(row["Category"], 0) / max(category_map.values()),
        row["Price"]
    ])

# Enhanced similarity with color compatibility and distance penalty
def compute_similarity(vec1, vec2, color1, color2):
    sim = 1 - cosine(vec1, vec2)

    # Color bonus
    if color2 in color_matching.get(color1, []):
        sim = min(sim + 0.05, 1.0)

    # Distance penalty
    penalty = np.linalg.norm(vec1 - vec2) * 0.2
    sim = max(0, sim - penalty)

    return sim

# Get recommendations for a user’s past purchases
def get_recommendations(purchased_ids, top_n=10):
    # Auto-wrap single integers
    if isinstance(purchased_ids, int):
        purchased_ids = [purchased_ids]

    if not purchased_ids:
        return []

    purchased_items = inventory[inventory["ID"].isin(purchased_ids)]
    if purchased_items.empty:
        return []

    purchased_vectors = [vectorize(row) for _, row in purchased_items.iterrows()]
    purchased_colors = [row["Color"] for _, row in purchased_items.iterrows()]

    recommendations = []

    for _, candidate in inventory.iterrows():
        if candidate["ID"] in purchased_ids:
            continue

        candidate_vector = vectorize(candidate)
        candidate_color = candidate["Color"]
        max_score = 0

        for pv, pc in zip(purchased_vectors, purchased_colors):
            score = compute_similarity(pv, candidate_vector, pc, candidate_color)
            if score > max_score:
                max_score = score

        recommendations.append({
            "id": candidate["ID"],
            "score": round(float(max_score), 4)  # for nice display
        })

    sorted_recs = sorted(recommendations, key=lambda x: x["score"], reverse=True)
    return sorted_recs[:top_n]
