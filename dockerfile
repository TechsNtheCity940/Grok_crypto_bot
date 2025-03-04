# Stage 1: Build TA-Lib
FROM ubuntu:22.04 as talib-builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download and install TA-Lib
WORKDIR /tmp
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install

# Stage 2: Final image
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu22.04

# Copy TA-Lib from builder stage
COPY --from=talib-builder /usr/include/ta-lib /usr/include/ta-lib
COPY --from=talib-builder /usr/lib/libta_lib* /usr/lib/

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --allow-change-held-packages \
    python3-pip \
    python3-dev \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA container toolkit
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add - && \
    curl -s -L https://nvidia.github.io/libnvidia-container/ubuntu20.04/libnvidia-container.list | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    nvidia-container-toolkit && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and add timeout settings
RUN pip3 install --upgrade pip && \
    pip3 config set global.timeout 600

# Copy requirements.txt
COPY requirements.txt .

# Install numpy first with specific version
RUN pip3 install --no-cache-dir numpy==1.21.6

# Install TA-Lib with specific version
RUN pip3 install --no-cache-dir TA-Lib==0.4.24

# Install TensorFlow separately
RUN pip3 install --no-cache-dir tensorflow

# Install PyTorch CPU version to avoid CUDA compatibility issues
RUN pip3 install --no-cache-dir torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional dependencies that might be missing
RUN pip3 install --no-cache-dir python-dotenv websocket-client

# Copy bot code, trained models, and environment variables
COPY . .
COPY models/trained_models/ ./models/trained_models/
COPY .env .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Run the bot with trade-only mode
CMD ["python3", "main.py", "--trade-only"]
