# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies, including latest TA-Lib wheel
RUN pip3 uninstall -y numpy && \
    pip3 install --no-cache-dir TA-Lib==0.4.32 && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy bot code  trained models
COPY . .
COPY models/trained_models/ ./models/trained_models/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the bot with trade-only mode
CMD ["python3", "main.py", "--trade-only"]