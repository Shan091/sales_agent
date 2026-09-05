from fastapi import APIRouter
from src.api.webhooks import router as webhooks_router
from src.api.assets import router as assets_router

api_router = APIRouter()
api_router.include_router(webhooks_router, prefix="/webhooks", tags=["webhooks"])
# Public asset served to Meta, which fetches attachment URLs itself — see src/api/assets.py.
api_router.include_router(assets_router, tags=["assets"])
