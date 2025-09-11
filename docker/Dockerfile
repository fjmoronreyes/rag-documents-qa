FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy project files (later we’ll refine this)
COPY ./src/pyproject.toml ./src/poetry.lock* /app/
RUN poetry install --no-root

# Default command (dummy for now)
CMD ["python", "--version"]