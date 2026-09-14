FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY ui ./ui
RUN uv sync --locked --no-dev

EXPOSE 8000 8501
CMD ["uvicorn", "magic_assistant.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
