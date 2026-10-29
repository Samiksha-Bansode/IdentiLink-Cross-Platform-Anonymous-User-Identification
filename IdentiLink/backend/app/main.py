from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.identify_routes import router as identify_router
from app.api.routes.fake_detection_routes import router as fake_router
from app.api.routes.trust_score_routes import router as trust_router

app = FastAPI(title="IdentiLink API", version="1.0.0")

# ✅ CORS — allows browser frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(identify_router)
app.include_router(fake_router)
app.include_router(trust_router)

@app.get("/")
def root():
    return {"message": "IdentiLink API is running ✅"}
