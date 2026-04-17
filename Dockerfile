# Use a slim Python image for efficiency
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories for persistence
RUN mkdir -p uploads reports

# Expose the port the app runs on
EXPOSE 8000

# Run the application with Gunicorn
# -w 1: reduced to one worker for 512MB RAM limits (Render Free)
# -k uvicorn.workers.UvicornWorker: use uvicorn workers for FastAPI
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "server:app", "--bind", "0.0.0.0:8000", "--timeout", "120"]
