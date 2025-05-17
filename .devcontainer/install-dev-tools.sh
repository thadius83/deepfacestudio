#!/bin/bash
# Script for setting up development tools in the GPU-enabled devcontainer

set -e

echo "Setting up DeepFace Studio GPU development environment"

# Install development tools
pip install --no-cache-dir black flake8 ipython jupyter pylint pytest streamlit

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
