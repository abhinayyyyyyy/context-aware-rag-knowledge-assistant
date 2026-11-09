# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

# Install runtime system packages (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and application directory
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ /app/src/
COPY app.py /app/app.py
COPY .env.example /app/.env.example

# Create storage directory and assign ownership to non-root user
RUN mkdir -p /app/data/storage \
    && chown -R appuser:appuser /app

USER appuser

# Expose Streamlit service port
EXPOSE 8501

# Healthcheck testing Streamlit core endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Launch Streamlit with container-friendly settings and configurable port
ENTRYPOINT ["sh", "-c", "exec streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
