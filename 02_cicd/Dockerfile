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

# Copy source code
COPY src/ ./src/

# Train on startup — in production replace with a serving entrypoint
ENTRYPOINT ["python", "src/train.py"]
