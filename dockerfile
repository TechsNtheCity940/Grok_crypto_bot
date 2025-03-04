# Fucking reliable Python 3.9 base image, no GPU nonsense
FROM python:3.9-slim

# Set the damn working directory
WORKDIR /app

# Install basic system shit, no build tools needed for prebuilt wheel
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements.txt
COPY requirements.txt .

# Install Python dependencies and a goddamn working TA-Lib wheel
RUN pip install --no-cache-dir numpy==1.21.0 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir https://files.pythonhosted.org/packages/39/4a/81383c7b7dc6692b5b434586038827dd595076a167ebfa96172e15d798bc/TA_Lib-0.4.28-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# Copy your bot code and trained models
COPY . .
COPY models/trained_models/ ./models/trained_models/

# Stop Python from buffering output like a jackass
ENV PYTHONUNBUFFERED=1

# Run the fucking bot already
CMD ["python3", "main.py", "--trade-only"]