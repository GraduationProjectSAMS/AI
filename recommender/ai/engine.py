import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from typing import Any, List, Dict

# File paths and sheet names
PIECES_FILE_PATH = "Data/Pieces.xlsx"
PURCHASES_FILE_PATH = "Data/Purchases.xlsx"
USERS_FILE_PATH = "Data/Users.xlsx"

PIECES_SHEET = "Pieces"
ORDER_ITEMS_SHEET = "Order Items"
USERS_SHEET = "Users"

# Column constants
ROOM_TYPE = "Room Type"
AESTHETIC = "Aesthetic"
CATEGORY = "Category"
PRICE = "Price"
COLOR = "Color"

# Color matching dictionary
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

def normalize(series: pd.Series) -> pd.Series:
    return (series - series.min()) / (series.max() - series.min())

def encode_dictionaries(inventory: pd.DataFrame):
    room = {val: i for i, val in enumerate(inventory[ROOM_TYPE].unique(), 1)}
    aesthetic = {val: i for i, val in enumerate(inventory[AESTHETIC].unique(), 1)}
    category = {val: i for i, val in enumerate(inventory[CATEGORY].unique(), 1)}
    return room, aesthetic, category

def vectorize_item(row, room_dict, aesthetic_dict, category_dict) -> np.ndarray:
    return np.array([
        float(room_dict.get(row.get(ROOM_TYPE), 0)) / max(room_dict.values()),
        float(aesthetic_dict.get(row.get(AESTHETIC), 0)) / max(aesthetic_dict.values()),
        float(category_dict.get(row.get(CATEGORY), 0)) / max(category_dict.values()),
        float(row[PRICE]) if pd.notna(row[PRICE]) else 0,
    ])

def compute_similarity(vector1, vector2, color1, color2) -> float:
    similarity = 1 - cosine(vector1, vector2)
    if color2 in color_matching.get(color1, []):
        similarity = min(similarity + 0.05, 1.0)
    similarity = max(0, similarity - np.linalg.norm(vector1 - vector2) * 0.2)
    return similarity

def get_user_purchase_history(users: pd.DataFrame, user_id: Any) -> List[int]:
    user = users[users["User ID"] == user_id]
    if user.empty:
        return []
    return eval(user.iloc[0]["Purchase History"])

def get_recommendations_for_user(user_id: Any) -> List[Dict[str, Any]]:
    # Load data
    inventory = pd.read_excel(PIECES_FILE_PATH, sheet_name=PIECES_SHEET)
    order_items = pd.read_excel(PURCHASES_FILE_PATH, sheet_name=ORDER_ITEMS_SHEET)
    users = pd.read_excel(USERS_FILE_PATH, sheet_name=USERS_SHEET)

    # Normalize prices
    inventory[PRICE] = normalize(inventory[PRICE])
    order_items["Unit Price"] = normalize(order_items["Unit Price"])

    # Encode attributes
    room_dict, aesthetic_dict, category_dict = encode_dictionaries(inventory)

    # Get user purchase history
    purchase_history = get_user_purchase_history(users, user_id)
    if not purchase_history:
        return []

    # Purchased items and vectors
    purchased_items = inventory[inventory["ID"].isin(purchase_history)]
    if purchased_items.empty:
        return []

    purchased_vectors = [vectorize_item(item, room_dict, aesthetic_dict, category_dict)
                         for _, item in purchased_items.iterrows()]
    purchased_colors = [item[COLOR] for _, item in purchased_items.iterrows()]

    # Build recommendation list
    recommendations = []
    for _, item in inventory.iterrows():
        if item["ID"] in purchase_history:
            continue

        item_vector = vectorize_item(item, room_dict, aesthetic_dict, category_dict)
        max_score = max(
            compute_similarity(item_vector, pv, pc, item[COLOR])
            for pv, pc in zip(purchased_vectors, purchased_colors)
        )

        recommendations.append((item["ID"], max_score))

    # Unique and sorted top 10
    unique_recs = {}
    for pid, score in recommendations:
        if pid not in unique_recs or unique_recs[pid] < score:
            unique_recs[pid] = score

    top_recs = sorted(unique_recs.items(), key=lambda x: x[1], reverse=True)[:10]
    return [{"product_id": pid, "score": round(score, 3)} for pid, score in top_recs]
