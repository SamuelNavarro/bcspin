# Use the official uv image as base
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app

# Create app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files and source package
COPY pyproject.toml uv.lock* README.md ./
COPY sproxxo/ ./sproxxo/

# Install dependencies
RUN uv sync --frozen --no-dev

# Development stage
FROM base as development

# Install development dependencies
RUN uv sync --frozen --extra dev

# Copy source code
COPY . .

# Production stage
FROM base as production

# Create non-root user
RUN groupadd -r sproxxo && useradd -r -g sproxxo sproxxo

# Copy source code
COPY . .

# Change ownership of the app directory
RUN chown -R sproxxo:sproxxo /app

# Switch to non-root user
USER sproxxo

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uv", "run", "uvicorn", "sproxxo.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
