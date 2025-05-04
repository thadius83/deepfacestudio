# DeepFace Studio DevContainer

This directory contains configuration for using DeepFace Studio in a development container with VS Code or DevPod.

## Setup Overview

The development container provides:

- CUDA 11.8 with cuDNN 8 for GPU acceleration
- Python 3.10 environment with all dependencies
- Convenient scripts for starting services
- Persistent volume for reference database
- Automatic mounting of model weights

## Prerequisites

- VS Code with Remote-Containers extension or DevPod
- Docker with NVIDIA Container Toolkit (for GPU support)

## Getting Started

1. Open the repository in VS Code or DevPod
2. When prompted, click "Reopen in Container"
3. The container will build and install all dependencies (this may take a few minutes)

## Development Workflow

Once the container is running, you can:

1. Start the FastAPI backend:
```bash
start-api
```

2. Start the Streamlit UI (in a new terminal):
```bash
start-ui
```

3. Access the UI at http://localhost:8501 and API at http://localhost:3900

Any code changes you make will automatically trigger hot-reloading of both services.

## Container Structure

- The entire repository is mounted at `/workspace`
- Reference database persists in a volume at `/data/reference_db`
- Model weights are mounted from `./deepface_weights` to `/root/.deepface/weights`
