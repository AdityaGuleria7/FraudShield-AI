# syntax=docker/dockerfile:1
FROM python:3.13-slim AS builder
WORKDIR /app
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Source only. Model artifacts are gitignored, so they are mounted at runtime
# (see docker-compose.yml) rather than copied; without one the dashboard still
# runs and the IEEE scoring endpoints return a clear 503.
COPY app/ app/
COPY scripts/ scripts/
COPY examples/ examples/

RUN useradd -m -u 1000 fraudshield \
    && mkdir -p var/audit artifacts \
    && chown -R fraudshield:fraudshield /app \
    && chmod 700 var/audit
USER fraudshield

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "app.dashboard:app", "--host", "0.0.0.0", "--port", "8000"]
