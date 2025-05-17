#!/bin/bash
# Script for setting up development tools in the GPU-enabled devcontainer

set -e

echo "Setting up DeepFace Studio GPU development environment"

# Install Python and system dependencies
apt-get update
apt-get install -y python3.10 python3.10-dev python3.10-distutils python3.10-venv python3-pip \
    build-essential libffi-dev \
    libjpeg-dev libpng-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libavfilter-dev libavformat-dev libavdevice-dev ffmpeg
update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install pip and upgrade it
python -m pip install --upgrade pip

# Install development tools
pip install --no-cache-dir black flake8 ipython jupyter pylint pytest streamlit

# Install TensorFlow and ML dependencies
pip install --no-cache-dir tensorflow opencv-python-headless numpy pillow pandas matplotlib
pip install --no-cache-dir fastapi uvicorn

# Make sure UI dependencies are installed
pip install --no-cache-dir -r /workspace/ui/requirements.txt

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

# Create the reference_db directory if it doesn't exist
mkdir -p /data/reference_db
chmod -R 777 /data/reference_db

# Check deepface weights directory
if [ ! -d "/root/.deepface/weights" ]; then
  mkdir -p /root/.deepface/weights
  chmod -R 777 /root/.deepface/weights
fi

echo "Development environment setup complete!"
