from fastapi import FastAPI, HTTPException
from engine import get_recommendations_endpoint, RecommendationRequest, RecommendationResponse
from typing import List, Optional

# uvicorn recommender.api.main:app --reload
app = FastAPI()


@app.get("/")
def root():
    return {"message": "Furnisique Recommendation API is running!"}


@app.post("/api/recommendations", response_model=List[RecommendationResponse])
async def recommend(request: RecommendationRequest):
    try:
        response = await get_recommendations_endpoint(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail="No purchased products found"
        )
