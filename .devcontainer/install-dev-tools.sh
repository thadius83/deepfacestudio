#!/bin/bash
# Script for installing development tools and dependencies

set -e

# Detect if we're in CPU mode
CPU_MODE=${DEEPFACE_CPU_MODE:-false}
echo "Setting up DeepFace Studio in $([ "$CPU_MODE" = "true" ] && echo "CPU" || echo "GPU") mode"

# update system
apt-get update
apt-get upgrade -y

# Install common dependencies
apt-get install -y software-properties-common wget curl git \
    build-essential libffi-dev \
    libjpeg-dev libpng-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libavfilter-dev libavformat-dev libavdevice-dev ffmpeg

# Install pip and upgrade it
python -m pip install --upgrade pip

# Install deepface requirements with appropriate backends
if [ "$CPU_MODE" = "true" ]; then
    # CPU-only installation 
    echo "Installing CPU-only dependencies"
    
    # Modify backend/requirements.txt to use CPU-only tensorflow if it exists
    if [ -f backend/requirements.txt ]; then
        # Replace tensorflow-gpu with tensorflow CPU version if present
        sed -i 's/tensorflow-gpu/tensorflow/g' backend/requirements.txt
    fi
    
    # Install with CPU-specific flags
    pip install --no-cache-dir -r backend/requirements.txt -r ui/requirements.txt
    
    # Configure DeepFace to use CPU-friendly models and detectors
    echo 'export DEEPFACE_DETECTOR="opencv"' >> /etc/bash.bashrc
    echo 'export MODEL_NAME="VGG-Face"' >> /etc/bash.bashrc
else
    # GPU installation (standard)
    echo "Installing GPU-enabled dependencies"
    pip install --no-cache-dir -r backend/requirements.txt -r ui/requirements.txt
fi

# Install development tools
pip install --no-cache-dir black flake8 ipython jupyter pylint pytest

# Create convenience scripts
mkdir -p /usr/local/bin/scripts
echo '#!/bin/bash
cd /workspace && uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 3900' > /usr/local/bin/scripts/start-api.sh

echo '#!/bin/bash
cd /workspace && streamlit run ui/streamlit_app.py --server.port=8501 --server.address=0.0.0.0' > /usr/local/bin/scripts/start-ui.sh

chmod +x /usr/local/bin/scripts/start-api.sh /usr/local/bin/scripts/start-ui.sh

# Add scripts to PATH and create aliases
echo 'export PATH="/usr/local/bin/scripts:${PATH}"' >> /etc/bash.bashrc
echo 'alias start-api="/usr/local/bin/scripts/start-api.sh"' >> /etc/bash.bashrc
echo 'alias start-ui="/usr/local/bin/scripts/start-ui.sh"' >> /etc/bash.bashrc

# Make sure the .deepface directory has proper permissions
mkdir -p /root/.deepface/weights
chmod -R 777 /root/.deepface

# Create the deepface_weights directory in workspace if it doesn't exist
mkdir -p /workspace/deepface_weights 
chmod -R 777 /workspace/deepface_weights

# If weights directory is not already a symlink, and it's empty, link it to the workspace dir
if [ ! -L /root/.deepface/weights ] && [ ! "$(ls -A /root/.deepface/weights)" ]; then
  rm -rf /root/.deepface/weights
  ln -s /workspace/deepface_weights /root/.deepface/weights
fi

# clean up
pip cache purge
apt-get autoremove -y
apt-get clean
