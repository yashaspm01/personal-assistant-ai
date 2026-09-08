import secrets
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader
from app.config import settings

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def require_api_key(key: str = Security(api_key_header)):
    """v1: single-user, single static key. Swap for JWT/OAuth when multi-user."""
    if not key or not secrets.compare_digest(key, settings.app_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True
