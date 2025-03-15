import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define constants
PIECES_FILE_PATH = "Data/Pieces.xlsx"
PURCHASES_FILE_PATH = "Data/Purchases.xlsx"

PIECES_SHEET = "Pieces"
ORDERS_TABLE_SHEET = "Orders Table"
ORDER_ITEMS_SHEET = "Order Items"

# Define attribute weights
weights = {
    "Room Type": 0.35,
    "Aesthetic": 0.25,
    "Category": 0.05,
    "Price": 0.15,
    "Color": 0.2  # Increased weight for color similarity
}

# Define a color compatibility dictionary
color_matching = {
    "Black": ["Gray", "White", "Silver"],
    "White": ["Beige", "Gray", "Black"],
    "Gray": ["Black", "White", "Silver"],
    "Brown": ["Beige", "Cream", "Tan"],
    "Blue": ["Light Blue", "Gray", "White"],
    "Green": ["Beige", "Brown", "White"],
    "Red": ["Beige", "Black", "Gold"],
    "Yellow": ["White", "Gray", "Gold"],
    "Pink": ["White", "Beige", "Light Gray"]
}

# Load inventory data
inventory = pd.read_excel(PIECES_FILE_PATH, sheet_name=PIECES_SHEET)

# Load purchases and orders data
orders = pd.read_excel(PURCHASES_FILE_PATH, sheet_name=ORDERS_TABLE_SHEET)
purchases = pd.read_excel(PURCHASES_FILE_PATH, sheet_name=ORDER_ITEMS_SHEET)

# Normalize numerical data (Price)
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

inventory["Price"] = normalize(inventory["Price"])
purchases["Unit Price"] = normalize(purchases["Unit Price"])

# Create encoding dictionaries
room_types = inventory["Room Type"].dropna().unique()
categories_list = inventory["Category"].dropna().unique()
aesthetics_list = inventory["Aesthetic"].dropna().unique()

room_types_dict = {room: idx + 1 for idx, room in enumerate(room_types)}
categories_dict = {category: idx + 1 for idx, category in enumerate(categories_list)}
aesthetics_dict = {aesthetic: idx + 1 for idx, aesthetic in enumerate(aesthetics_list)}

# Function to vectorize a furniture item
def vectorize_item(row, last_color):
    max_room = max(room_types_dict.values())
    max_aesthetic = max(aesthetics_dict.values())
    max_category = max(categories_dict.values())

    # Check if the color is the same or complementary
    color_similarity = 1 if row["Color"] == last_color else (0.8 if row["Color"] in color_matching.get(last_color, []) else 0.5)

    return np.array([
        room_types_dict.get(row["Room Type"], 0) / max_room,
        aesthetics_dict.get(row["Aesthetic"], 0) / max_aesthetic,
        categories_dict.get(row["Category"], 0) / max_category,
        float(row["Price"]),
        color_similarity
    ])

# Compute similarity score
def compute_similarity(vector1, vector2, weights):
    weight_array = np.array([weights["Room Type"], weights["Aesthetic"], weights["Category"], weights["Price"], weights["Color"]])
    distance = np.sqrt(np.sum(weight_array * (vector1 - vector2) ** 2))
    return 1 / (1 + distance)  # Convert distance to similarity

# Get last purchase of the user (assuming User ID = 1)
user_orders = orders[orders["User ID"] == 1]
if user_orders.empty:
    print("No purchases found for User ID = 1.")
    exit()

latest_order_id = user_orders.sort_values(by="Order Date", ascending=False).iloc[0]["Order ID"]
last_purchases = purchases[purchases["Order ID"] == latest_order_id]

if last_purchases.empty:
    print("No items found for the last order.")
    exit()

print(f"Latest Order ID: {latest_order_id}")
print("Last Purchased Items:")
print(last_purchases)

# Get all previously purchased product IDs
previous_purchases = purchases[purchases["Order ID"].isin(user_orders["Order ID"])]["Product ID"].unique()

# Compute recommendations based on the last purchase
recommendations = []
for _, purchased_item in last_purchases.iterrows():
    product_id = purchased_item["Product ID"]
    product_details = inventory[inventory["ID"] == product_id]
    
    if not product_details.empty:
        purchased_vector = vectorize_item(product_details.iloc[0], product_details.iloc[0]["Color"])
        last_price = purchased_item["Unit Price"]

        for _, inventory_item in inventory.iterrows():
            if inventory_item["ID"] not in previous_purchases:  # Exclude previously bought items
                
                # Apply price filtering (±20% price range)
                if abs(inventory_item["Price"] - last_price) > 0.2:
                    continue  # Skip items outside budget range
                
                inventory_vector = vectorize_item(inventory_item, product_details.iloc[0]["Color"])
                score = compute_similarity(inventory_vector, purchased_vector, weights)
                recommendations.append((inventory_item["ID"], score))

# Sort and remove duplicates
unique_recommendations = {}
for product_id, score in recommendations:
    if product_id not in unique_recommendations or unique_recommendations[product_id] < score:
        unique_recommendations[product_id] = score

ranked_products = sorted(unique_recommendations.items(), key=lambda x: x[1], reverse=True)

# Display top recommendations
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
    plt.title('Top 10 Product Recommendations (With Color & Budget Filtering)')
    plt.tight_layout()
    plt.show()
else:
    print("No recommendations found.")
