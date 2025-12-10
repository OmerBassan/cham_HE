# Chameleon: LLM Robustness Benchmark Framework
# Docker image for running evaluations

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Create Projects directory
RUN mkdir -p /app/Projects

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command - show help
CMD ["python", "cli.py", "--help"]

# Labels
LABEL org.opencontainers.image.title="Chameleon"
LABEL org.opencontainers.image.description="LLM Robustness Benchmark Framework"
LABEL org.opencontainers.image.url="https://github.com/stevesolun/Chameleon"
LABEL org.opencontainers.image.source="https://github.com/stevesolun/Chameleon"
LABEL org.opencontainers.image.authors="Steve Solun"

