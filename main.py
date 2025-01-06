import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cosine

# Load inventory data
inventory = pd.read_excel("Data/Pieces.xlsx")
purchases = pd.read_excel("Data/Purchases.xlsx")

# Attribute weights
weights = {
    "Room Type": 0.4,
    "Aesthetic": 0.3,
    "Category": 0.05,
    "Price": 0.15,
    "Color": 0.1
}

def vectorize_item(row):
    room_types = {"Living Room": 1, "Kitchen": 2, "Bedroom": 3}
    aesthetics = {"Modern": 1, "Classic": 2, "Contemporary": 3}
    categories = {"Seats": 1, "Tables": 2, "Beds": 3}
    colors = {"Gray": 1, "White": 2, "Brown": 3, "Black": 4}
    
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

# Reevaluate Weighting to ensure weights properly affect the similarity:
# Convert weights to fractions to sum to 1
total_weight = sum(weights.values())
weights = {k: v / total_weight for k, v in weights.items()}


def compute_similarity(vector1, vector2, weights):
    weight_array = np.array([weights["Room Type"], weights["Aesthetic"], weights["Category"], weights["Price"], weights["Color"]])
    vector1 = np.array(vector1, dtype=float)
    vector2 = np.array(vector2, dtype=float)
    
    # Compute weighted Euclidean distance
    distance = np.sqrt(np.sum(weight_array * (vector1 - vector2) ** 2))
    return 1 / (1 + distance)  # Convert distance to similarity


# Get user purchases
user_purchases = purchases[purchases["User ID"] == 1]["ID"]

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# Normalize numerical data
inventory["Price"] = normalize(inventory["Price"])
purchases["Price"] = normalize(purchases["Price"])

# Recommendations
recommendations = []
for _, inventory_item in inventory.iterrows():
    inventory_vector = vectorize_item(inventory_item)
    for purchased_item in user_purchases:
        purchased_vector = vectorize_item(inventory[inventory["ID"] == purchased_item].iloc[0])
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

product_ids = [item[0] for item in ranked_products]
scores = [item[1] for item in ranked_products]

# Visualization
product_ids = [item[0] for item in ranked_products]
scores = [item[1] for item in ranked_products]

plt.figure(figsize=(10, 6))
plt.bar(product_ids[:10], scores[:10], color='skyblue')
plt.xlabel('Product ID')
plt.ylabel('Similarity Score')
plt.title('Top 10 Product Recommendations')
plt.show()