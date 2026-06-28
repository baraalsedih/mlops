# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder (keeps the final image lean)
COPY --from=builder /install /usr/local

# Copy source code and tests
COPY src/ ./src/
COPY tests/ ./tests/

# CMD is overridable at runtime; ENTRYPOINT would not be
CMD ["python", "src/train.py"]
