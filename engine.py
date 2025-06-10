import pandas as pd
import numpy as np
from numpy.random import Generator, PCG64
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine

# Define constants for file paths and sheet names
PIECES_FILE_PATH = "Data/Pieces.xlsx"
PURCHASES_FILE_PATH = "Data/Purchases.xlsx"
USERS_FILE_PATH = "Data/Users.xlsx"

PIECES_SHEET = "Pieces"
ROOMS_SHEET = "Rooms"
CATEGORIES_SHEET = "Categories"
AESTHETICS_SHEET = "Aesthetics"
ORDER_ITEMS_SHEET = "Order Items"
ORDERS_TABLE_SHEET = "Orders Table"
USERS_SHEET = "Users"

# Define constants for attribute names
ROOM_TYPE = "Room Type"
AESTHETIC = "Aesthetic"
CATEGORY = "Category"
PRICE = "Price"
COLOR = "Color"

# Define a color compatibility dictionary with prioritized rankings
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

# Load data
inventory = pd.read_excel(PIECES_FILE_PATH, sheet_name=PIECES_SHEET)
order_items = pd.read_excel(PURCHASES_FILE_PATH, sheet_name=ORDER_ITEMS_SHEET)
orders_table = pd.read_excel(PURCHASES_FILE_PATH, sheet_name=ORDERS_TABLE_SHEET)
users = pd.read_excel(USERS_FILE_PATH, sheet_name=USERS_SHEET)

# Attribute weights
weights = {
    ROOM_TYPE: 0.4,
    AESTHETIC: 0.3,
    CATEGORY: 0.05,
    PRICE: 0.1,
    COLOR: 0.15
}

# Initialize a random generator (do this once at module level)
rng = Generator(PCG64(seed=42))

# Normalize numerical data (Price)
scaler = MinMaxScaler()
inventory[[PRICE]] = scaler.fit_transform(inventory[[PRICE]])
order_items[["Unit Price"]] = scaler.fit_transform(order_items[["Unit Price"]])

# Create encoding dictionaries for categorical values
room_types_dict = {room: idx for idx, room in enumerate(inventory[ROOM_TYPE].unique(), start=1)}
aesthetics_dict = {aesthetic: idx for idx, aesthetic in enumerate(inventory[AESTHETIC].unique(), start=1)}
categories_dict = {category: idx for idx, category in enumerate(inventory[CATEGORY].unique(), start=1)}

def color_score(purchased_color, inventory_color):
    compatible = color_matching.get(purchased_color, [])
    if inventory_color == purchased_color:
        return 1.0
    elif inventory_color in compatible[:3]:
        return 0.9
    elif inventory_color in compatible[:6]:
        return 0.7
    elif inventory_color in compatible:
        return 0.5
    else:
        return 0.0

# Vectorize item with encoded categorical values
def weighted_vectorize(row):
    return np.array([
        room_types_dict.get(row.get(ROOM_TYPE), 0) / max(room_types_dict.values()) * weights[ROOM_TYPE],
        aesthetics_dict.get(row.get(AESTHETIC), 0) / max(aesthetics_dict.values()) * weights[AESTHETIC],
        categories_dict.get(row.get(CATEGORY), 0) / max(categories_dict.values()) * weights[CATEGORY],
        float(row[PRICE]) * weights[PRICE] if pd.notna(row[PRICE]) else 0,
    ])

# Enhanced Similarity Calculation with Distance Penalty
def compute_similarity(vector1, vector2, purchased_color, inventory_color):
    vector1 = np.array(vector1, dtype=float)
    vector2 = np.array(vector2, dtype=float)
    
    # Base similarity (0-0.80 range)
    similarity = (1 - cosine(vector1, vector2)) * 0.80
    
    # Color compatibility (0-0.08 range)
    color_comp = color_score(purchased_color, inventory_color) * 0.08
    
    # Room type bonus (0-0.07 range)
    room_bonus = 0.07 if vector1[0] == vector2[0] else 0
    
    # Use a tiny fraction of another attribute as tie-breaker
    tie_breaker = vector2[2] * 0.0001  # e.g., use category index
    total_similarity = similarity + color_comp + room_bonus + tie_breaker
    return np.clip(total_similarity, 0, 1.0)  # Cleaner than min/max

# Core function: get recommendations based on last purchase
def get_recommendations(user_id, top_n=10):
    # Get user information
    user_row = users[users["User ID"] == user_id]
    if user_row.empty:
        raise ValueError(f"User ID {user_id} not found.")
    
    # Get complete purchase history
    purchase_history = eval(user_row.iloc[0]["Purchase History"])
    if not purchase_history:
        return []

    # Get all items from the last purchase
    last_purchase_items = inventory[inventory["ID"].isin(purchase_history)]
    
    # Create vectors for all last purchased items
    last_vectors = []
    last_colors = []
    last_rooms = []
    for _, item in last_purchase_items.iterrows():
        last_vectors.append(weighted_vectorize(item))
        last_colors.append(item["Color"])
        last_rooms.append(item[ROOM_TYPE])

    recommendations = []
    for _, candidate in inventory.iterrows():
        if candidate["ID"] in purchase_history:
            continue
            
        candidate_vector = weighted_vectorize(candidate)
        candidate_color = candidate["Color"]
        
        max_score = 0
        for last_vec, last_col in zip(last_vectors, last_colors):
            similarity = compute_similarity(last_vec, candidate_vector, last_col, candidate_color)
            if similarity > max_score:
                max_score = similarity
        
        recommendations.append({
            "id": candidate["ID"],
            "compatibility_score": round(float(max_score), 4)
        })
        
    # Sort by score and return top N recommendations
    recommendations = sorted(recommendations, key=lambda x: x["compatibility_score"], reverse=True)
    return recommendations[:top_n]