# XFIN Docker Image
# Multi-stage build for smaller production image
# Supports both Streamlit dashboard and FastAPI modes

# Stage 1: Build dependencies
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production image
FROM python:3.10-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Install runtime dependencies (for matplotlib, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Build argument to select mode: "streamlit" or "api"
ARG APP_MODE=streamlit
ENV APP_MODE=${APP_MODE}

# Expose ports (8501 for Streamlit, 8000 for FastAPI)
EXPOSE 8501 8000

# Health check (works for both modes)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${APP_MODE:+$([ "$APP_MODE" = "api" ] && echo 8000 || echo 8501)}/health || exit 1

# Default command: run based on APP_MODE
CMD if [ "$APP_MODE" = "api" ]; then \
    uvicorn api.main:app --host 0.0.0.0 --port 8000; \
    else \
    streamlit run XFIN/stress_app.py --server.port=8501 --server.address=0.0.0.0; \
    fi

