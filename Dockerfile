# ---------- Dependencies Stage (Cached) ----------
FROM python:3.10-slim AS deps

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
  
# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with better caching
# This layer will be cached unless requirements.txt changes
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- Production Stage ----------
FROM python:3.10-slim AS production

# Set working directory
WORKDIR /app

# Copy virtual environment from deps stage
COPY --from=deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install python-dotenv
RUN pip install python-dotenv

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PORT=8000 \
    WORKERS=17 \
    TIMEOUT=180 \
    KEEP_ALIVE=45 \
    LOG_LEVEL=info

# Copy configuration files first (they change less frequently)
COPY gunicorn.conf.py .
COPY config.py .
COPY pyproject.toml .

# Copy the main application files
COPY main.py .
COPY admin_interface.py .

# Copy the ingestion_retrieval module
COPY ingestion_retrieval/ ./ingestion_retrieval/

# Create required directories and set permissions
RUN mkdir -p uploaded_folders data logs && \
    chmod -R 755 /app && \
    chmod -R 755 /app/uploaded_folders && \
    chmod -R 755 /app/data && \
    chmod -R 755 /app/logs

# Set non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Command to run the application
CMD ["gunicorn", "-c", "gunicorn.conf.py", "main:app"]