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
OLD_PRICE = "Old Price"

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

# Incorporate discounts for boosting recommendations
def calculate_discount(row):
    if pd.notna(row[OLD_PRICE]):
        return (row[OLD_PRICE] - row[PRICE]) / row[OLD_PRICE]
    return 0

inventory["Discount"] = inventory.apply(calculate_discount, axis=1)

# Create encoding dictionaries for categorical values
room_types_dict = {room: idx for idx, room in enumerate(inventory[ROOM_TYPE].unique(), start=1)}
aesthetics_dict = {aesthetic: idx for idx, aesthetic in enumerate(inventory[AESTHETIC].unique(), start=1)}
categories_dict = {category: idx for idx, category in enumerate(inventory[CATEGORY].unique(), start=1)}

# Vectorize item with encoded categorical values
def vectorize_item(row):
    return np.array([
        float(room_types_dict.get(row.get(ROOM_TYPE), 0)) / max(room_types_dict.values()),
        float(aesthetics_dict.get(row.get(AESTHETIC), 0)) / max(aesthetics_dict.values()),
        float(categories_dict.get(row.get(CATEGORY), 0)) / max(categories_dict.values()),
        float(row[PRICE]) if pd.notna(row[PRICE]) else 0,
        float(row["Discount"] if pd.notna(row["Discount"]) else 0)
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
    purchase_history = eval(user["Purchase History"])  # Convert string to list
    if not purchase_history:
        print(f"No purchase history found for User ID {user_id}.")
        continue

    # Get details of all last purchased items
    purchased_items = inventory[inventory["ID"].isin(purchase_history)]
    if purchased_items.empty:
        print(f"No valid purchases found for User ID {user_id}.")
        continue

    # Vectorize all purchased items and store their colors
    purchased_vectors = []
    purchased_colors = []
    for _, purchased_item in purchased_items.iterrows():
        purchased_vectors.append(vectorize_item(purchased_item))
        purchased_colors.append(purchased_item[COLOR])

    # Generate recommendations by comparing against ALL purchased items
    recommendations = []
    for _, inventory_item in inventory.iterrows():
        if inventory_item["ID"] in purchase_history:
            continue  # Skip already purchased items

        inventory_vector = vectorize_item(inventory_item)
        max_score = 0  # Track the highest similarity score across all purchased items

        for purchased_vector, purchased_color in zip(purchased_vectors, purchased_colors):
            score = compute_similarity(
                inventory_vector, 
                purchased_vector, 
                purchased_color, 
                inventory_item[COLOR]
            )
            if score > max_score:
                max_score = score  # Keep the highest score

        recommendations.append((inventory_item["ID"], max_score))

    # Rank and sort recommendations
    unique_recommendations = {}
    for product_id, score in recommendations:
        if product_id not in unique_recommendations or unique_recommendations[product_id] < score:
            unique_recommendations[product_id] = score

    ranked_products = sorted(unique_recommendations.items(), key=lambda x: x[1], reverse=True)

    # Display results for the user
    top_n_recommendations = ranked_products[:10]
    print(f"\nTop Recommendations for User ID {user_id}:")
    print(top_n_recommendations)

    # Visualization
    if ranked_products:
        product_ids = [item[0] for item in ranked_products]
        scores = [item[1] for item in ranked_products]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(product_ids[:10])), scores[:10], color='skyblue', tick_label=product_ids[:10])
        plt.xlabel('Product ID')
        plt.ylabel('Similarity Score')
        plt.title(f'Top 10 Recommendations for User ID {user_id}')
        plt.tight_layout()
        plt.show()
    else:
        print(f"No recommendations found for User ID {user_id}.")