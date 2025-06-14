import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Normalize numerical data (Price)
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

inventory[PRICE] = normalize(inventory[PRICE])
order_items["Unit Price"] = normalize(order_items["Unit Price"])

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
    similarity = 1 - cosine(vector1, vector2)

    # Color Bonus — Reduced to Avoid Overcompensation
    if inventory_color in color_matching.get(purchased_color, []):
        similarity = min(similarity + 0.05, 1.0)
        
    # Distance Penalty for More Precision in Similarity
    distance_penalty = np.linalg.norm(vector1 - vector2) * 0.2
    similarity = max(0, similarity - distance_penalty)

    return similarity

# Process each user's last purchases
for _, user in users.iterrows():
    user_id = user["User ID"]
    purchase_history = eval(user["Purchase History"])

    if not purchase_history:
        print(f"No purchase history for User ID {user_id}")
        continue

    # --- Get Last Purchased Item Only ---
    last_purchase_id = purchase_history[-1]
    last_purchased_item = inventory[inventory["ID"] == last_purchase_id]

    if last_purchased_item.empty:
        print(f"Last purchased item not found for User ID {user_id}")
        continue

    last_vector = weighted_vectorize(last_purchased_item.iloc[0])
    last_color = last_purchased_item.iloc[0][COLOR]

    recommendations = []
    for _, candidate in inventory.iterrows():
        if candidate["ID"] in purchase_history:
            continue

        candidate_vector = weighted_vectorize(candidate)
        base_similarity = 1 - cosine(last_vector, candidate_vector)
        base_similarity = max(0, base_similarity)  # Handle NaN

        # Apply color influence
        color_weight = color_score(last_color, candidate[COLOR])
        final_score = base_similarity * 0.85 + color_weight * 0.15

        recommendations.append((candidate["ID"], final_score))

    top_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]

    print(f"\nTop Recommendations for User ID {user_id}:")
    print(top_recommendations)

    # Plot Results
    if top_recommendations:
        product_ids = [item[0] for item in top_recommendations]
        scores = [item[1] for item in top_recommendations]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(product_ids)), scores, tick_label=product_ids, color='skyblue')
        plt.title(f'Top 10 Recommendations for User {user_id}')
        plt.xlabel('Product ID')
        plt.ylabel('Similarity Score')
        plt.tight_layout()
        plt.show()
    else:
        print(f"No recommendations found for User ID {user_id}.")