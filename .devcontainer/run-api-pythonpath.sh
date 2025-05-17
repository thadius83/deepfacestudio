#!/bin/bash
# Script to run the API with PYTHONPATH set to include the workspace
# This should help with Python module resolution issues

# Set PYTHONPATH to current directory
export PYTHONPATH=/workspace

# Run the API with the module name
cd /workspace
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 3900
