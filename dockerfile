# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies including TA-Lib requirements
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    build-essential \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Download and install TA-Lib C library
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir numpy && \
    pip3 install --no-cache-dir TA-Lib==0.4.32 && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy bot code  trained models
COPY . .
COPY models/trained_models/ ./models/trained_models/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the bot with trade-only mode
CMD ["python3", "main.py", "--trade-only"]
