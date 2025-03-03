# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and TA-Lib wheel file
COPY requirements.txt .
COPY ta_lib-0.6.0-cp312-cp312-win_amd64.whl .

# Install Python dependencies, excluding talib-binary from PyPI
RUN pip3 install --no-cache-dir -r requirements.txt

# Install TA-Lib from the local wheel file
RUN pip3 install --no-cache-dir ta_lib-0.6.0-cp312-cp312-win_amd64.whl

# Copy bot code and trained models
COPY . .
COPY models/trained_models/ ./models/trained_models/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the bot with trade-only mode
CMD ["python3", "main.py", "--trade-only"]