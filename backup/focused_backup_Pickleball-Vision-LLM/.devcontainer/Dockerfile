FROM continuumio/miniconda3

# Set working directory
WORKDIR /workspace

# Set shell to bash
SHELL ["/bin/bash", "-c"]

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Initialize conda
RUN conda init bash

# Set environment variables (if needed)
ENV PYTHONUNBUFFERED=1

# Create and activate conda environment (optional)
# RUN conda create -n myenv python=3.9 \
#     && echo "conda activate myenv" >> ~/.bashrc

# Copy requirements.txt if you have one
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# Set default command
CMD ["/bin/bash"]
