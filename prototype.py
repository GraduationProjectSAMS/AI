import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Define constants for file paths and sheet names
PIECES_FILE_PATH = "Data/Pieces.xlsx"
PURCHASES_FILE_PATH = "Data/Purchases.xlsx"

PIECES_SHEET = "Pieces"
ROOMS_SHEET = "Rooms"
CATEGORIES_SHEET = "Categories"
AESTHETICS_SHEET = "Aesthetics"
ORDER_ITEMS_SHEET = "Order Items"
ORDERS_TABLE_SHEET = "Orders Table"

# Define constants for attribute names
ROOM_TYPE = "Room Type"
AESTHETIC = "Aesthetic"
CATEGORY = "Category"
PRICE = "Price"
COLOR = "Color"
OLD_PRICE = "Old Price"  # Define constant for "Old Price"

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

# Load inventory and purchase data
inventory = pd.read_excel(PIECES_FILE_PATH, sheet_name=PIECES_SHEET)
order_items = pd.read_excel(PURCHASES_FILE_PATH, sheet_name=ORDER_ITEMS_SHEET)
orders_table = pd.read_excel(PURCHASES_FILE_PATH, sheet_name=ORDERS_TABLE_SHEET)

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

# Improved Vectorization with Better Scaling
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

# Fetch recent purchases for User ID 1
user_orders = orders_table[orders_table["User ID"] == 1].sort_values(by="Timestamp", ascending=False)
last_order_id = user_orders.iloc[0]["Order ID"]
user_purchases = order_items[order_items["Order ID"] == last_order_id]

# Create vectors for purchased items
purchased_vectors = []
purchased_product_ids = set()
purchased_colors = {}
for _, purchased_item in user_purchases.iterrows():
    product_id = purchased_item["Product ID"]
    purchased_product_ids.add(product_id)
    product_details = inventory[inventory["ID"] == product_id]
    if not product_details.empty:
        purchased_vectors.append(vectorize_item(product_details.iloc[0]))
        purchased_colors[product_id] = product_details.iloc[0][COLOR]

# Generate recommendations efficiently with .merge()
recommendations = []
inventory = inventory[~inventory["ID"].isin(purchased_product_ids)]

for _, inventory_item in inventory.iterrows():
    inventory_vector = vectorize_item(inventory_item)
    for purchased_vector, purchased_color in zip(purchased_vectors, purchased_colors.values()):
        score = compute_similarity(inventory_vector, purchased_vector, purchased_color, inventory_item[COLOR])
        recommendations.append((inventory_item["ID"], score))

# Rank and sort
unique_recommendations = {}
for product_id, score in recommendations:
    if product_id not in unique_recommendations or unique_recommendations[product_id] < score:
        unique_recommendations[product_id] = score

ranked_products = sorted(unique_recommendations.items(), key=lambda x: x[1], reverse=True)

# Display results
top_n_recommendations = ranked_products[:10]
print("Top Recommendations:", top_n_recommendations)

# Visualization
if ranked_products:
    product_ids = [item[0] for item in ranked_products]
    scores = [item[1] for item in ranked_products]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(product_ids[:10])), scores[:10], color='skyblue', tick_label=product_ids[:10])
    plt.xlabel('Product ID')
    plt.ylabel('Similarity Score')
    plt.title('Top 10 Product Recommendations')
    plt.tight_layout()
    plt.show()
else:
    print("No recommendations found.")