# DeepFace Studio DevContainer

This directory contains configuration for using DeepFace Studio in a development container with VS Code. The setup supports both CPU and GPU modes.

## Usage

### Prerequisites

- VS Code with the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension
- Docker and Docker Compose
- For GPU mode: NVIDIA GPU with drivers and nvidia-container-toolkit installed

### Opening in DevContainer

1. Clone the repository
2. Open the folder in VS Code
3. When prompted, click "Reopen in Container" or run the "Remote-Containers: Reopen in Container" command

### Switching Between CPU and GPU Modes

The default mode is CPU. To switch to GPU mode:

1. Set the `RUNTIME_MODE` environment variable to `gpu` before starting the container:

```bash
# For Linux/macOS
export RUNTIME_MODE=gpu
code .

# For Windows PowerShell
$env:RUNTIME_MODE="gpu"
code .
```

2. Alternatively, modify the `devcontainer.json` file to change the default:

```json
"containerEnv": {
  "RUNTIME_MODE": "gpu"
}
```

## Development Workflow

Once inside the container, you can:

1. Start the backend API:
```bash
start-api
```

2. Start the Streamlit UI in a new terminal:
```bash
start-ui
```

3. Access the UI at http://localhost:8501 and the API at http://localhost:3900

Any changes you make to the code will automatically reload both services.
