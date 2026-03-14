from fastapi import APIRouter

from app.api.routes import fraud_router

api_router = APIRouter()
api_router.include_router(fraud_router)
