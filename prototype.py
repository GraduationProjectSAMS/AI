import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

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
CATEGORY = "Category"  # Updated to match the column name in the Pieces sheet
PRICE = "Price"
COLOR = "Color"

# Load inventory data and additional sheets
try:
    inventory = pd.read_excel(PIECES_FILE_PATH, sheet_name=PIECES_SHEET)
    rooms = pd.read_excel(PIECES_FILE_PATH, sheet_name=ROOMS_SHEET)
    categories = pd.read_excel(PIECES_FILE_PATH, sheet_name=CATEGORIES_SHEET)
    aesthetics = pd.read_excel(PIECES_FILE_PATH, sheet_name=AESTHETICS_SHEET)
except ValueError as e:
    print(f"Error loading sheets: {e}")
    print("Please ensure the following sheets exist in 'Pieces.xlsx':")
    print(f"- {ROOMS_SHEET}")
    print(f"- {CATEGORIES_SHEET}")
    print(f"- {AESTHETICS_SHEET}")
    exit(1)

# Extract unique values from sheets
room_types = rooms[ROOM_TYPE].dropna().unique()
categories_list = categories["Categories"].dropna().unique()  # Use "Categories" for the Categories sheet
aesthetics_list = aesthetics[AESTHETIC].dropna().unique()

# Create dictionaries for encoding
room_types_dict = {room: idx + 1 for idx, room in enumerate(room_types)}
categories_dict = {category: idx + 1 for idx, category in enumerate(categories_list)}
aesthetics_dict = {aesthetic: idx + 1 for idx, aesthetic in enumerate(aesthetics_list)}

# Load purchases data (use the Order Items sheet)
purchases = pd.read_excel(PURCHASES_FILE_PATH, sheet_name=ORDER_ITEMS_SHEET)

# Attribute weights
weights = {
    ROOM_TYPE: 0.4,
    AESTHETIC: 0.3,
    CATEGORY: 0.05,
    PRICE: 0.15,
    COLOR: 0.1
}

# Normalize numerical data (Price)
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# Normalize prices in inventory and purchases
inventory[PRICE] = normalize(inventory[PRICE])
purchases["Unit Price"] = normalize(purchases["Unit Price"])

# Vectorize an item based on its attributes
def vectorize_item(row):
    max_room_type = max(room_types_dict.values())
    max_aesthetic = max(aesthetics_dict.values())
    max_category = max(categories_dict.values())
    
    return np.array([
        room_types_dict.get(row[ROOM_TYPE], 0) / max_room_type,
        aesthetics_dict.get(row[AESTHETIC], 0) / max_aesthetic,
        categories_dict.get(row[CATEGORY], 0) / max_category,
        float(row[PRICE]),  # Already normalized
        1  # Placeholder for color (you can add color logic if needed)
    ])

# Compute similarity between two vectors
def compute_similarity(vector1, vector2, weights):
    weight_array = np.array([weights[ROOM_TYPE], weights[AESTHETIC], weights[CATEGORY], weights[PRICE], weights[COLOR]])
    vector1 = np.array(vector1, dtype=float)
    vector2 = np.array(vector2, dtype=float)
    
    # Compute weighted Euclidean distance
    distance = np.sqrt(np.sum(weight_array * (vector1 - vector2) ** 2))
    return 1 / (1 + distance)  # Convert distance to similarity

# Get user purchases (assuming User ID = 1)
user_orders = pd.read_excel(PURCHASES_FILE_PATH, sheet_name=ORDERS_TABLE_SHEET)
user_order_ids = user_orders[user_orders["User ID"] == 1]["Order ID"]
user_purchases = purchases[purchases["Order ID"].isin(user_order_ids)]

# Debug: Check if user_purchases is empty
if user_purchases.empty:
    print("No purchases found for User ID = 1.")
else:
    print("User Purchases:")
    print(user_purchases)

# Precompute vectors for purchased items
purchased_vectors = []
for _, purchased_item in user_purchases.iterrows():
    product_id = purchased_item["Product ID"]
    product_details = inventory[inventory["ID"] == product_id]
    if not product_details.empty:
        purchased_vectors.append(vectorize_item(product_details.iloc[0]))
    else:
        print(f"Product ID {product_id} not found in inventory.")

# Debug: Check if purchased_vectors is empty
if not purchased_vectors:
    print("No valid purchased items found for recommendations.")
else:
    print(f"Found {len(purchased_vectors)} purchased items for recommendations.")

# Generate recommendations
recommendations = []
for _, inventory_item in inventory.iterrows():
    inventory_vector = vectorize_item(inventory_item)
    for purchased_vector in purchased_vectors:
        score = compute_similarity(inventory_vector, purchased_vector, weights)
        recommendations.append((inventory_item["ID"], score))

# Sort recommendations
unique_recommendations = {}
for product_id, score in recommendations:
    if product_id not in unique_recommendations or unique_recommendations[product_id] < score:
        unique_recommendations[product_id] = score

ranked_products = sorted(unique_recommendations.items(), key=lambda x: x[1], reverse=True)

# Display results
top_n_recommendations = ranked_products[:10]
print("Top Recommendations:", top_n_recommendations)

# Visualization
if ranked_products:  # Check if there are recommendations
    product_ids = [item[0] for item in ranked_products]
    scores = [item[1] for item in ranked_products]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(product_ids[:10])), scores[:10], color='skyblue', tick_label=product_ids[:10])
    plt.xlabel('Product ID')
    plt.ylabel('Similarity Score')
    plt.title('Top 10 Product Recommendations')
    plt.tight_layout()  # Adjust layout to prevent label overlap
    plt.show()
else:
    print("No recommendations found.")