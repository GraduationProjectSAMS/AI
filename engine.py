#Modular Functions

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler

# Load data once
inventory = pd.read_excel("Data/Pieces.xlsx", sheet_name="Pieces")

# Normalization
scaler = MinMaxScaler()
inventory["Price"] = scaler.fit_transform(inventory[["Price"]])

# Encode categorical attributes
def encode_column(series):
    unique_vals = series.unique()
    mapping = {val: i for i, val in enumerate(unique_vals)}
    return series.map(mapping), mapping

inventory["Room Encoded"], room_map = encode_column(inventory["Room Type"])
inventory["Aesthetic Encoded"], aesthetic_map = encode_column(inventory["Aesthetic"])
inventory["Category Encoded"], category_map = encode_column(inventory["Category"])
inventory["Color Encoded"], color_map = encode_column(inventory["Color"])

# Vectorization
def vectorize(row):
    return np.array([
        row["Room Encoded"] / max(room_map.values()),
        row["Aesthetic Encoded"] / max(aesthetic_map.values()),
        row["Category Encoded"] / max(category_map.values()),
        row["Price"],
        row["Color Encoded"] / max(color_map.values())
    ])

# Similarity function
def compute_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

# Main recommendation function
def get_recommendations(item_id, top_n=5):
    try:
        target_row = inventory[inventory["ID"] == item_id]
        if target_row.empty:
            return []

        target_vector = vectorize(target_row.iloc[0])

        similarities = []
        for _, row in inventory.iterrows():
            if row["ID"] == item_id:
                continue  # skip same item

            candidate_vector = vectorize(row)
            score = compute_similarity(target_vector, candidate_vector)
            similarities.append((row["ID"], score))

        sorted_results = sorted(similarities, key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_results[:top_n]]

    except Exception:
        return []
