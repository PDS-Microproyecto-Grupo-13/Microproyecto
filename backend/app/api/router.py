from fastapi import APIRouter

from app.api.routes import health

api_router = APIRouter()

# Register endpoint route groups
api_router.include_router(health.router)

# Future modules can be attached easily:
# api_router.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
# api_router.include_router(models.router, prefix="/models", tags=["Models"])
