from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from recommender.ai.engine import get_recommendations_for_user

app = FastAPI()

class UserRequest(BaseModel):
    user_id: int

@app.post("/recommendations")
def recommend(user: UserRequest):
    try:
        recommendations = get_recommendations_for_user(user.user_id)
        return {"user_id": user.user_id, "recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Furnisique Recommendation API is running!"}
