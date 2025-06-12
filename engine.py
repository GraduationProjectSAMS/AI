from typing import List, Optional

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from numpy.random import Generator, PCG64
from pydantic import BaseModel
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler

# FastAPI app initialization
app = FastAPI(title="Recommendation Service")

# API Configuration
API_BASE_URL = "https://furnisique.servehttp.com/api"
PRODUCTS_ENDPOINT = f"{API_BASE_URL}/products?per_page=500"
ORDERS_ENDPOINT = f"{API_BASE_URL}/purchase-history"

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


# Pydantic models for API responses
class RecommendationRequest(BaseModel):
    top_n: Optional[int] = 10
    token: str  # Bearer token passed in payload


class RecommendationResponse(BaseModel):
    id: int
    compatibility_score: float


# Global variables for data storage
inventory = None
scaler = None
room_types_dict = None
aesthetics_dict = None
categories_dict = None
rng = None

# Attribute weights
weights = {
    ROOM_TYPE: 0.4,
    AESTHETIC: 0.3,
    CATEGORY: 0.05,
    PRICE: 0.1,
    COLOR: 0.15
}


# Fetch products from API
async def fetch_products_from_api(token: str):
    """Fetch products from the API and convert to DataFrame format"""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.get(PRODUCTS_ENDPOINT, headers=headers)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch products")

        products_data = response.json()

        # Convert API data to DataFrame format matching your existing structure
        # Adjust these field mappings based on your actual API response structure
        products_df = pd.DataFrame([{
            "ID": product.get("id"),
            "Room Type": product.get("rooms"),
            "Aesthetic": product.get("aesthetic"),
            "Category": product.get("category"),
            "Price": product.get("price"),
            "Color": product.get("color"),
            # Add other fields as needed
        } for product in products_data.get("data", [])])

        return products_df


# Fetch orders from API
async def fetch_purchasedProducts_from_api(token: str):
    """Fetch orders from the API and convert to DataFrame format"""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.get(ORDERS_ENDPOINT, headers=headers)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch orders")

        purchasedProducts = response.json()

        # purchased_product_ids
        return purchasedProducts.get("data", [])


# Initialize data from APIs
async def initialize_data_from_api(token: str):
    """Initialize all data from APIs"""
    global inventory, scaler
    global room_types_dict, aesthetics_dict, categories_dict, rng

    try:
        # Fetch products from API
        inventory = await fetch_products_from_api(token)
        # Fetch user's purchase history
        purchased_product_ids = await fetch_purchasedProducts_from_api(token)

        # Initialize random generator
        rng = Generator(PCG64(seed=42))

        # Normalize numerical data (Price)
        scaler = MinMaxScaler()
        if not inventory.empty and PRICE in inventory.columns:
            inventory[[PRICE]] = scaler.fit_transform(inventory[[PRICE]])

        # Create encoding dictionaries for categorical values
        room_types_dict = {room: idx for idx, room in enumerate(inventory[ROOM_TYPE].unique(), start=1)}
        aesthetics_dict = {aesthetic: idx for idx, aesthetic in enumerate(inventory[AESTHETIC].unique(), start=1)}
        categories_dict = {category: idx for idx, category in enumerate(inventory[CATEGORY].unique(), start=1)}

        return True, purchased_product_ids
    except Exception as e:
        print(f"Error initializing data from API: {e}")
        return False, []


# Initialize data from Excel files (fallback)
def initialize_data_from_files():
    """Initialize data from Excel files (original method) - simplified for fallback"""
    global inventory, scaler
    global room_types_dict, aesthetics_dict, categories_dict, rng

    try:
        # Load only inventory data for fallback
        inventory = pd.read_excel(PIECES_FILE_PATH, sheet_name=PIECES_SHEET)

        # Initialize random generator
        rng = Generator(PCG64(seed=42))

        # Normalize numerical data (Price)
        scaler = MinMaxScaler()
        inventory[[PRICE]] = scaler.fit_transform(inventory[[PRICE]])

        # Create encoding dictionaries for categorical values
        room_types_dict = {room: idx for idx, room in enumerate(inventory[ROOM_TYPE].unique(), start=1)}
        aesthetics_dict = {aesthetic: idx for idx, aesthetic in enumerate(inventory[AESTHETIC].unique(), start=1)}
        categories_dict = {category: idx for idx, category in enumerate(inventory[CATEGORY].unique(), start=1)}

        return True
    except Exception as e:
        print(f"Error loading data from files: {e}")
        return False


# YOUR ORIGINAL FUNCTIONS (UNCHANGED)
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
def get_recommendations_for_user(purchase_history, top_n=10):
    """Modified version that doesn't need user_id"""
    if not purchase_history:
        return []

    # Get all items from the purchase history
    last_purchase_items = inventory[inventory["ID"].isin(purchase_history)]

    # Create vectors for all purchased items
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


# FastAPI Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize data on startup"""
    # Try to initialize from local files on startup
    success = initialize_data_from_files()

    if not success:
        print("Failed to initialize data from files on startup!")


@app.get("/")
async def root():
    return {"message": "Recommendation Service API", "version": "1.0"}


@app.post("/api/recommendations", response_model=List[RecommendationResponse])
async def get_recommendations_endpoint(request: RecommendationRequest):
    """Get recommendations for the authenticated user"""
    try:
        # Load fresh data from API with the provided token
        success, purchased_product_ids = await initialize_data_from_api(request.token)
        if not success:
            raise HTTPException(status_code=503, detail="Failed to fetch data from API")

        # Get recommendations based on the user's purchase history
        recommendations = get_recommendations_for_user(purchased_product_ids, request.top_n)

        return [
            RecommendationResponse(
                id=rec["id"],
                compatibility_score=rec["compatibility_score"]
            )
            for rec in recommendations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "data_loaded": inventory is not None
    }


# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
