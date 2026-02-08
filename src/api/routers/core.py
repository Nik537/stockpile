"""Core routes for the Stockpile API (root and health check)."""

from api.schemas import HealthResponse, RootResponse
from fastapi import APIRouter

router = APIRouter(tags=["Core"])


@router.get(
    "/",
    response_model=RootResponse,
    summary="API root",
    description="Returns API name and version.",
)
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "Stockpile API", "version": "1.0.0"}


@router.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns server health status.",
)
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
