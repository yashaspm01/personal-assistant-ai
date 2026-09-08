"""Structured logging: console + file, with request-level tracing via middleware."""
import logging
import sys
import time
import uuid
from fastapi import Request


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("app/data/app.log"),
        ],
    )


async def request_logging_middleware(request: Request, call_next):
    logger = logging.getLogger("request")
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    logger.info(f"[{request_id}] {request.method} {request.url.path} — started")
    try:
        response = await call_next(request)
        duration_ms = round((time.time() - start) * 1000, 1)
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} — "
            f"{response.status_code} in {duration_ms}ms"
        )
        response.headers["x-request-id"] = request_id
        return response
    except Exception as e:
        duration_ms = round((time.time() - start) * 1000, 1)
        logger.error(
            f"[{request_id}] {request.method} {request.url.path} — "
            f"UNHANDLED EXCEPTION after {duration_ms}ms: {e}"
        )
        raise
