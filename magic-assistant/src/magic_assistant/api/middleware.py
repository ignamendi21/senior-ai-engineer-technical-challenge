import logging
import re
import time
from uuid import uuid4

from fastapi import Request, Response

logger = logging.getLogger("magic_assistant.api")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


def request_id(value: str | None) -> str:
    if value and _REQUEST_ID_RE.fullmatch(value):
        return value
    return str(uuid4())


async def observe_request(request: Request, call_next) -> Response:
    correlation_id = request_id(request.headers.get("X-Request-ID"))
    request.state.request_id = correlation_id
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = correlation_id
        return response
    finally:
        latency_ms = (time.perf_counter() - started) * 1_000
        logger.info(
            "request_complete request_id=%s method=%s path=%s status=%s latency_ms=%.2f",
            correlation_id,
            request.method,
            request.url.path,
            status_code,
            latency_ms,
        )
