import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from magic_assistant.agent.runtime import (
    AssistantRuntime,
    LiveAssistantRuntime,
    RuntimeInitializationError,
)
from magic_assistant.api.middleware import observe_request
from magic_assistant.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ReadinessResponse,
    build_chat_response,
)

logger = logging.getLogger("magic_assistant.api")
RuntimeFactory = Callable[[], AssistantRuntime]


def create_app(runtime_factory: RuntimeFactory = LiveAssistantRuntime.create) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.runtime = None
        application.state.runtime_error = None
        try:
            try:
                application.state.runtime = runtime_factory()
            except RuntimeInitializationError as error:
                application.state.runtime_error = str(error)
                logger.error("runtime_initialization_failed error_type=%s", type(error).__name__)
            except Exception as error:
                application.state.runtime_error = "Assistant runtime initialization failed."
                logger.error("runtime_initialization_failed error_type=%s", type(error).__name__)
            yield
        finally:
            runtime = application.state.runtime
            if runtime is not None:
                try:
                    runtime.close()
                except Exception as error:
                    logger.error("runtime_shutdown_failed error_type=%s", type(error).__name__)

    application = FastAPI(
        title="Magic Assistant Demo API",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.middleware("http")(observe_request)

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        details = [{"location": list(item["loc"]), "type": item["type"]} for item in error.errors()]
        return JSONResponse(status_code=422, content={"detail": details})

    @application.exception_handler(Exception)
    async def unexpected_exception_handler(request: Request, error: Exception) -> JSONResponse:
        correlation_id = getattr(request.state, "request_id", "unavailable")
        logger.error(
            "unexpected_request_failure request_id=%s error_type=%s",
            correlation_id,
            type(error).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": correlation_id},
            headers={"X-Request-ID": correlation_id},
        )

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @application.get("/ready", response_model=ReadinessResponse)
    def ready(request: Request, response: Response) -> ReadinessResponse:
        runtime = request.app.state.runtime
        if runtime is None or not runtime.ready:
            response.status_code = 503
            detail = request.app.state.runtime_error or "Assistant runtime is unavailable."
            return ReadinessResponse(status="unavailable", detail=detail)
        return ReadinessResponse(status="ready")

    @application.post("/api/chat", response_model=ChatResponse)
    def chat(chat_request: ChatRequest, request: Request) -> ChatResponse:
        runtime = request.app.state.runtime
        if runtime is None or not runtime.ready:
            raise HTTPException(status_code=503, detail="Assistant runtime is unavailable.")
        try:
            state = runtime.invoke(chat_request.message, thread_id=chat_request.thread_id)
            response = build_chat_response(
                state,
                thread_id=chat_request.thread_id,
                request_id=request.state.request_id,
            )
        except Exception as error:
            logger.error(
                "chat_failure request_id=%s error_type=%s",
                request.state.request_id,
                type(error).__name__,
            )
            raise HTTPException(status_code=500, detail="Internal server error") from error
        categories = [error.category for error in state.get("errors", [])]
        logger.info(
            "chat_complete request_id=%s intent=%s route=%s errors=%s",
            request.state.request_id,
            response.intent.value,
            "->".join(response.route_trace),
            ",".join(categories) or "none",
        )
        return response

    return application


app = create_app()
