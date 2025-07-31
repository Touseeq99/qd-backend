# ---------- Base Layer for Heavy Libraries ----------
FROM python:3.10-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages with specific versions
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    https://download.pytorch.org/whl/cpu/torch-2.1.0%2Bcpu-cp310-cp310-linux_x86_64.whl && \
    pip install --no-cache-dir sentence-transformers==2.6.1
    
    
    # ---------- Dependencies Stage ----------
    FROM base AS deps
    
    # Copy requirements.txt
    COPY requirements.txt .
    
    # Install all remaining dependencies
    RUN pip install --no-cache-dir --retries 5 --timeout 300 -r requirements.txt
    
    
    # ---------- Production Stage ----------
    FROM python:3.10-slim AS production
    
    WORKDIR /app
    
    COPY --from=deps /opt/venv /opt/venv
    ENV PATH="/opt/venv/bin:$PATH"
    
    # Reinstall python-dotenv (if not in requirements.txt)
    RUN pip install python-dotenv
    
    # Install runtime dependencies only
    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        && rm -rf /var/lib/apt/lists/*
    
    # Environment variables
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PYTHONHASHSEED=random \
        PYTHONMALLOC=malloc \
        MALLOC_TRIM_THRESHOLD_=65536 \
        PIP_NO_CACHE_DIR=off \
        PIP_DISABLE_PIP_VERSION_CHECK=on \
        PIP_DEFAULT_TIMEOUT=100 \
        PORT=8000 \
        WORKERS=8 \
        TIMEOUT=300 \
        KEEP_ALIVE=60 \
        LOG_LEVEL=info \
        HF_HOME=/home/appuser/.cache/huggingface \
        NLTK_DATA=/app/nltk_data \
        OMP_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        VECLIB_MAXIMUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1
    
    # Copy configs
    COPY gunicorn.conf.py .
    COPY config.py .
    COPY pyproject.toml .
    
    # Copy all necessary files
COPY . .

# Ensure all Python files are included
RUN find . -name "*.py" -type f -print
    
    # Create non-root user and set up directories
RUN groupadd -r appuser && \
    useradd -r -g appuser appuser && \
    mkdir -p /app/uploaded_folders \
             /app/data \
             /app/logs \
             /home/appuser/.cache/huggingface \
             /app/ingestion_retrieval \
             /app/nltk_data && \
    chmod -R 755 /app && \
    chown -R appuser:appuser /app /home/appuser/.cache /app/nltk_data && \
    chmod -R 777 /tmp

    
# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Switch to non-root user
USER appuser

# Command to run the application
CMD ["gunicorn", "-c", "gunicorn.conf.py", "main:app"]
    