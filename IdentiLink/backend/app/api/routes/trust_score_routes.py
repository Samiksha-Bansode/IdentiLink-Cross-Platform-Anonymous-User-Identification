from fastapi import APIRouter
from pydantic import BaseModel
from app.services.trust_score_service import trust_score

router = APIRouter()

class Username(BaseModel):
    username: str

@router.post("/trust_score")
def score(data: Username):
    return trust_score(data.username)
