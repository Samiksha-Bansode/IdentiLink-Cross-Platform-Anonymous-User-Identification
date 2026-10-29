from fastapi import APIRouter
from pydantic import BaseModel
from app.services.identity_service import identify_user

router = APIRouter()

class Username(BaseModel):
    username: str

@router.post("/identify")
def identify(data: Username):
    return identify_user(data.username)
