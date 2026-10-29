from fastapi import APIRouter
from pydantic import BaseModel
from app.services.fake_detection_service import detect_fake

router = APIRouter()

class Username(BaseModel):
    username: str

@router.post("/fake_detection")
def fake(data: Username):
    return detect_fake(data.username)
