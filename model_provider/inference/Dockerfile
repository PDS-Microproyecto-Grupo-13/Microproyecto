FROM python:3.12.14-slim

# Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install curl for diagnostic/healthchecks and libgomp1 for LightGBM OpenMP support
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install pinned dependencies
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt

# Copy operational start script and healthcheck
COPY start.py healthcheck.py ./

EXPOSE 5001 5002

CMD ["python", "start.py"]
