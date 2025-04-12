# Furnisique AI Recommender System

Welcome to the AI component of **Furnisique** — a smart furniture recommendation system designed to enhance the user's shopping experience by offering tailored furniture suggestions based on similarity.

---

## 🧠 Project Overview
This system analyzes a user's last purchase(s) and computes similarity-based furniture recommendations using a hybrid logic of encoded features and cosine similarity. It's deployed as a RESTful Flask API that integrates smoothly with a Laravel backend.

---

## 🔧 Tech Stack
| Layer            | Technology      |
|------------------|-----------------|
| Data Processing  | Python, Pandas, NumPy |
| Similarity Logic | Scipy (Cosine Similarity) |
| API Layer        | Flask           |
| Backend Integration | Laravel (calls the Flask API) |

---

## 🔍 How It Works
1. **Vector Encoding**: Each product is converted into a vector based on room type, aesthetic, category, normalized price, and color encoding.
2. **Similarity Computation**: Uses cosine similarity to compare the vector of a new item with a user's last purchases.
3. **Color Compatibility Boost**: Adds a minor boost to scores when color aesthetics match.
4. **Top-N Ranking**: Returns top-N product recommendations (excluding purchased items).

---

## 🚀 Running the API
### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Start the Flask Server
```bash
python recommender_api.py
```

### 3. Example API Request
```
GET http://localhost:5000/recommendations?user_id=12&top_n=10
```

### ✅ Sample Response
```json
{
  "user_id": 12,
  "recommendations": [
    {"id": 5, "score": 0.92},
    {"id": 14, "score": 0.89},
    {"id": 3, "score": 0.87}
  ]
}
```

---

## 🧪 Testing
Run the local tests to validate the model:
```bash
python tests/test_recommender.py
```

---

## 👥 Contributors
- AI Engineer: Nour Maged
