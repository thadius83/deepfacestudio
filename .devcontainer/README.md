# DeepFace Studio GPU Development Container

This directory contains configuration files for setting up a GPU-enabled development environment for DeepFace Studio using VS Code's Remote - Containers extension.

## Requirements

- Docker Desktop or Docker Engine with NVIDIA container toolkit installed
- VS Code with the Remote - Containers extension
- NVIDIA GPU with compatible drivers
- Docker Compose (for running both UI and API services)

## Features

- Uses the same backend Dockerfile as the main application (TensorFlow GPU enabled)
- GPU acceleration via NVIDIA runtime
- Pre-configured for Python linting and formatting
- Volume mounting for persistent reference database
- Persistent storage for model weights
- Development tools (black, flake8, pylint, etc.)
- Auto-start scripts for API and UI

## Usage

1. Open this project in VS Code
2. Click the "Remote Explorer" icon in the activity bar
3. Select "Reopen in Container" from the Remote menu
4. Wait for the container to build and initialize

## Running the Application

Once inside the development container, you can start the services using:

```bash
# Start the API service
start-api

# In a new terminal inside the container
start-ui
```

The API will be available at http://localhost:3900 and the UI at http://localhost:8501.

## GPU Acceleration

This development container is configured to use NVIDIA GPU acceleration. The application will automatically detect and use available GPU resources for face detection and recognition tasks.

## Troubleshooting

If you encounter GPU-related issues:

1. Ensure NVIDIA drivers are properly installed on your host system
2. Verify that nvidia-container-toolkit is installed and configured
3. Check the output of `nvidia-smi` on your host to confirm GPU availability
4. Ensure you have sufficient GPU memory for the operations being performed
