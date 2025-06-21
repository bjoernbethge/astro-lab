FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create astro-lab user
RUN groupadd --gid 1000 astro-lab \
    && useradd --uid 1000 --gid astro-lab --shell /bin/bash --create-home astro-lab

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.8 /uv /uvx /bin/

# Set working directory and give ownership to astro-lab user
WORKDIR /app
RUN chown -R astro-lab:astro-lab /app

# Copy project files and set ownership
COPY --chown=astro-lab:astro-lab pyproject.toml uv.lock .marimo.toml ./

# Create MLflow directories with proper ownership
RUN mkdir -p /mlflow/db /mlflow/artifacts /mlflow/models /mlflow/experiments /mlflow/runs /mlflow/prompts /snippets\
    && chown -R astro-lab:astro-lab /mlflow

# Switch to astro-lab user for dependency installation
USER astro-lab

# install packages
RUN uv add marimo-flow --no-sync # marimo-flow is marimo + mlflow
RUN uv sync --no-dev

RUN uv pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# Copy startup script
COPY --chown=astro-lab:astro-lab start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose ports
EXPOSE 2718 5000

# Default command (can be overridden by docker-compose)
CMD ["/app/start.sh"]

