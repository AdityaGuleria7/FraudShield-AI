# syntax=docker/dockerfile:1
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build deps
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY app/ app/
COPY artifacts/ artifacts/
COPY scripts/ scripts/

# Create audit directory with restrictive permissions
RUN mkdir -p var/audit && chmod 700 var/audit

# Non-root user for security
RUN useradd -m -u 1000 fraudshield
RUN chown -R fraudshield:fraudshield /app
USER fraudshield

# Expose dashboard port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "app.dashboard:app", "--host", "0.0.0.0", "--port", "8000"]
