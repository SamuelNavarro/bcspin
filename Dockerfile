# Use the official uv image as base
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache \
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
COPY artifacts/ ./artifacts/
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

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

# Create non-root user with home directory
RUN groupadd -r sproxxo && useradd -r -g sproxxo -m -d /home/sproxxo sproxxo

# Copy source code
COPY . .

# Ensure entrypoint script has execute permissions
RUN chmod +x entrypoint.sh

# Create cache directory and change ownership
RUN mkdir -p /tmp/uv-cache && \
    chown -R sproxxo:sproxxo /app /home/sproxxo /tmp/uv-cache

# Switch to non-root user
USER sproxxo

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Default command
CMD ["./entrypoint.sh"]
