# DeepFace Studio DevContainer

This directory contains configuration for using DeepFace Studio in a development container with VS Code or DevPod.

## Setup Options

Two container configurations are available:

1. **GPU Version** (branch: `feature/devcontainer`) - For development with CUDA GPU acceleration **forthcoming**
2. **CPU Version** (branch: `feature/devcontainer-cpu`) - For development on machines without GPUs

## Setup Overview

The development container provides:

- Python 3.10 environment with all dependencies
- Convenient scripts for starting services
- Persistent volume for reference database
- Automatic configuration of model weights

## Prerequisites

### For GPU Version
- VS Code with Remote-Containers extension or DevPod
- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with appropriate drivers

### For CPU Version
- VS Code with Remote-Containers extension or DevPod
- Docker

## Getting Started

1. Choose the appropriate branch:
   ```bash
   # For GPU support
   git checkout feature/devcontainer
   
   # For CPU-only (no GPU required)
   git checkout feature/devcontainer-cpu
   ```

2. Open the repository in VS Code or DevPod
3. When prompted, click "Reopen in Container"
4. The container will build and install all dependencies (this may take a few minutes)

## Development Workflow

Once the container is running, you can:

### Method 1: Using convenience scripts

1. Start the FastAPI backend:
```bash
start-api
```

2. Start the Streamlit UI (in a new terminal):
```bash
start-ui
```



2. Access the UI at http://localhost:8501 and API at http://localhost:3900

Any code changes you make will automatically trigger hot-reloading of both services.

## Performance Expectations

- **CPU Version**: Will work but with reduced performance; uses more CPU-friendly settings
