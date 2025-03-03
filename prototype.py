import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

# Load inventory data
inventory = pd.read_excel("Data/Pieces.xlsx")

# Load purchases data (use the Order Items sheet)
purchases = pd.read_excel("Data/Purchases.xlsx", sheet_name="Order Items")

# Attribute weights
weights = {
    "Room Type": 0.4,
    "Aesthetic": 0.3,
    "Category": 0.05,
    "Price": 0.15,
    "Color": 0.1
}

# Normalize numerical data (Price)
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# Normalize prices in inventory and purchases
inventory["Price"] = normalize(inventory["Price"])
purchases["Unit Price"] = normalize(purchases["Unit Price"])

# Vectorize an item based on its attributes
def vectorize_item(row):
    room_types = {"Living": 1, "Kitchen": 2, "Bedroom": 3, "Office": 4}
    aesthetics = {"Modern": 1, "Classic": 2, "Cozy": 3, "Vintage": 4, "Stylish": 5, "Rustic": 6, "Industrial": 7, "Minimalist": 8, "Functional": 9}
    categories = {"Seats": 1, "Tables": 2, "Beds": 3, "Storage": 4, "Entertainment": 5}
    colors = {"Gray": 1, "Brown": 2, "Blue": 3, "Mahogany": 4, "White": 5, "Black": 6, "Clear": 7, "Beige": 8, "Walnut": 9, "Cherry": 10, "Teal": 11, "Natural": 12}
    
    max_room_type = max(room_types.values())
    max_aesthetic = max(aesthetics.values())
    max_category = max(categories.values())
    max_color = max(colors.values())
    
    return np.array([
        room_types.get(row["Room Type"], 0) / max_room_type,
        aesthetics.get(row["Aesthetic"], 0) / max_aesthetic,
        categories.get(row["Category"], 0) / max_category,
        float(row["Price"]),  # Already normalized
        colors.get(row["Color"], 0) / max_color
    ])

# Compute similarity between two vectors
def compute_similarity(vector1, vector2, weights):
    weight_array = np.array([weights["Room Type"], weights["Aesthetic"], weights["Category"], weights["Price"], weights["Color"]])
    vector1 = np.array(vector1, dtype=float)
    vector2 = np.array(vector2, dtype=float)
    
    # Compute weighted Euclidean distance
    distance = np.sqrt(np.sum(weight_array * (vector1 - vector2) ** 2))
    return 1 / (1 + distance)  # Convert distance to similarity

# Get user purchases (assuming User ID = 1)
user_orders = pd.read_excel("Data/Purchases.xlsx", sheet_name="Orders Table")
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