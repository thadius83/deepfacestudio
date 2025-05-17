FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

# Install system dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y software-properties-common wget curl git \
    build-essential libffi-dev \
    libjpeg-dev libpng-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libavfilter-dev libavformat-dev libavdevice-dev ffmpeg

# Install Python 3.10
RUN apt-get install -y python3.10 python3.10-dev python3.10-distutils python3.10-venv python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install pip and upgrade it
RUN python -m pip install --upgrade pip

# Create necessary directories
RUN mkdir -p /data/reference_db && chmod -R 777 /data/reference_db && \
    mkdir -p /root/.deepface/weights && chmod -R 777 /root/.deepface

WORKDIR /workspace

# Install TensorFlow GPU and common ML dependencies
RUN pip install --no-cache-dir tensorflow==2.15.0 opencv-python-headless numpy pillow 

# Pre-install other common dependencies
RUN pip install --no-cache-dir fastapi uvicorn streamlit pandas matplotlib

# Note: Actual project dependencies will be installed by the post-create script
