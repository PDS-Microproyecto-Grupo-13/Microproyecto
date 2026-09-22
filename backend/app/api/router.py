from fastapi import APIRouter

from app.api.routes import analytics, health, predictions

api_router = APIRouter()

# Register endpoint route groups
api_router.include_router(health.router)
api_router.include_router(predictions.router)
api_router.include_router(analytics.router)

# Future modules can be attached easily:
# api_router.include_router(models.router, prefix="/models", tags=["Models"])
