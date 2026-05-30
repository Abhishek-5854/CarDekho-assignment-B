FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Install Python deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port (Railway/Platforms set $PORT at runtime)
EXPOSE 8000

# Use shell form so $PORT is expanded at container runtime by the shell
# Railway provides $PORT at runtime; default to 8000 if not set.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
