from fastapi import FastAPI, HTTPException
from recommender.ai.engine import get_recommendations_for_user

#uvicorn recommender.api.main:app --reload
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Furnisique Recommendation API is running!"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int):
    try:
        recommendations = get_recommendations_for_user(user_id)
        return {"user_id": user_id, "recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
