# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies, including build tools for TA-Lib
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    wget \
    build-essential \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies, including TA-Lib Python wrapper
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir TA-Lib

# Copy bot code and trained models
COPY . .
COPY models/trained_models/ ./models/trained_models/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/lib:/usr/local/lib

# Run the bot with trade-only mode
CMD ["python3", "main.py", "--trade-only"]