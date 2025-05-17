#!/bin/bash
# Script for installing development tools and dependencies for GPU setup

set -e

echo "Setting up DeepFace Studio in GPU mode"

# update system
apt-get update
apt-get upgrade -y

# Install common dependencies and Python
apt-get install -y software-properties-common wget curl git \
    build-essential libffi-dev \
    libjpeg-dev libpng-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libavfilter-dev libavformat-dev libavdevice-dev ffmpeg

# Install Python 3.10
apt-get install -y python3.10 python3.10-dev python3.10-distutils python3.10-venv python3-pip
update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install pip and upgrade it
python -m pip install --upgrade pip

# Install dependencies
echo "Installing GPU-enabled dependencies"
pip install --no-cache-dir -r backend/requirements.txt -r ui/requirements.txt

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
