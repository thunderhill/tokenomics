# syntax=docker/dockerfile:1

# Builder: resolve the locked dependency set with uv, isolated from the runtime image.
FROM python:3.12-slim AS builder

# Installed from PyPI rather than copied from the astral-sh/uv image: some build
# environments (this one included) can reach PyPI but not ghcr.io. No `--mount=type=cache`
# on the RUN steps below, deliberately: it needs BuildKit's buildx plugin, which is not
# guaranteed present (this sandbox doesn't have it) -- correctness over faster rebuilds.
RUN pip install --no-cache-dir uv~=0.5.0

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

# Dependencies first, so the layer only invalidates when pyproject.toml/uv.lock change --
# not on every source edit.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY README.md ./
# --no-editable: the runtime stage below copies only .venv, not ./src. A default
# (editable) uv sync links the venv to ./src via a .pth file, which resolves to
# nothing once that directory is gone -- ModuleNotFoundError at container start.
RUN uv sync --frozen --no-dev --no-editable

# Runtime: no uv, no build tools, no source outside the venv.
FROM python:3.12-slim AS runtime

RUN useradd --create-home --uid 1000 tokenomics
WORKDIR /app
COPY --from=builder --chown=tokenomics:tokenomics /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    TOKENOMICS_DATABASE_URL="postgresql://tokenomics:tokenomics@postgres:5432/tokenomics"

USER tokenomics
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=5 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["tokenomics", "serve"]
