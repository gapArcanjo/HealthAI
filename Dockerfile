# ── Build Stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Download spaCy Portuguese model
RUN python -m spacy download pt_core_news_lg

# ── Production Stage ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS production

WORKDIR /app

# Security: non-root user
RUN groupadd -r healthai && useradd -r -g healthai healthai

# Copy from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=healthai:healthai . .

# Create directories for models and logs
RUN mkdir -p ml/models ml/reports && chown -R healthai:healthai ml/

USER healthai

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]