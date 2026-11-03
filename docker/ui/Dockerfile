FROM python:3.13-slim-trixie AS build

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Dependencies first, so source changes do not invalidate this layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable --group ui

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --group ui


FROM python:3.13-slim-trixie

RUN useradd --create-home --uid 1000 app
COPY --from=build --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src/scilitrag/ui/app.py /app/app.py
COPY --chown=app:app .streamlit /app/.streamlit
ENV PATH="/app/.venv/bin:$PATH"
USER app
WORKDIR /app

EXPOSE 8501
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
