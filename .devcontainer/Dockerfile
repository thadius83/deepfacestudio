# Select base image based on runtime mode
ARG RUNTIME_MODE=cpu

# Use CUDA base image for GPU, standard Python for CPU
FROM ${RUNTIME_MODE:cpu=python:3.10-bullseye:+nvidia/cuda:11.8.0-runtime-ubuntu22.04}

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# For CUDA based image, we need to install Python
RUN if [ "$RUNTIME_MODE" = "gpu" ]; then \
    apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-venv python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*; \
    fi

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install development tools
RUN pip install --no-cache-dir \
    black \
    flake8 \
    ipython \
    jupyter \
    pylint \
    pytest

# Create directory for application code
WORKDIR /workspace

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Make sure the .deepface directory has proper permissions
RUN mkdir -p /root/.deepface/weights && \
    chmod -R 777 /root/.deepface

# Create scripts directory for convenience commands
RUN mkdir -p /usr/local/bin/scripts

# Create startup scripts
RUN echo '#!/bin/bash\ncd /workspace && uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 3900' > /usr/local/bin/scripts/start-api.sh && \
    echo '#!/bin/bash\ncd /workspace && streamlit run ui/streamlit_app.py --server.port=8501 --server.address=0.0.0.0' > /usr/local/bin/scripts/start-ui.sh && \
    chmod +x /usr/local/bin/scripts/start-api.sh /usr/local/bin/scripts/start-ui.sh

# Add scripts to PATH
ENV PATH="/usr/local/bin/scripts:${PATH}"

# Create alias commands for convenient access
RUN echo 'alias start-api="/usr/local/bin/scripts/start-api.sh"' >> /etc/bash.bashrc && \
    echo 'alias start-ui="/usr/local/bin/scripts/start-ui.sh"' >> /etc/bash.bashrc

# Set RUNTIME_MODE environment variable
ENV RUNTIME_MODE=${RUNTIME_MODE:-cpu}

# Add conditional configuration for CPU mode
RUN if [ "$RUNTIME_MODE" = "cpu" ]; then \
    # Add CPU-specific configurations here if needed \
    echo "Configuring for CPU mode"; \
    fi

# Set working directory
WORKDIR /workspace
