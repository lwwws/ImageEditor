#!/bin/bash

# Exit immediately on error
set -e

ENV_DIR="venv-edit-image"
echo "Installing edit-image in editable mode..."

# If the venv doesn't exist, ask which Python to use and create it
if [ ! -d "$ENV_DIR" ]; then
    read -p "Virtual environment doesn't exist. Enter the Python executable to use for creating it [default: python3]: " PYTHON_EXEC
    PYTHON_EXEC=${PYTHON_EXEC:-python3}

    if ! command -v "$PYTHON_EXEC" &> /dev/null; then
        echo "Error: '$PYTHON_EXEC' not found. Install it or specify another Python version."
        exit 1
    fi

    if ! "$PYTHON_EXEC" -m venv --help &> /dev/null; then
        echo "Error: '$PYTHON_EXEC' does not support venv. Install the appropriate python-venv package."
        exit 1
    fi

    echo "Creating virtual environment in $ENV_DIR using $PYTHON_EXEC..."
    "$PYTHON_EXEC" -m venv "$ENV_DIR"
else
    echo "Using existing virtual environment in $ENV_DIR."
fi

# Activate the venv
if [ ! -f "$ENV_DIR/bin/activate" ]; then
    echo "Error: '$ENV_DIR/bin/activate' not found. Virtual environment may be incomplete."
    exit 1
fi

echo "Activating virtual environment..."
source "$ENV_DIR/bin/activate"

# Install dependencies
pip install --upgrade pip
pip install -e .

echo
echo "Setup complete."
echo
echo "To activate the virtual environment, run:"
echo "    source $ENV_DIR/bin/activate"
echo
echo "Then you can run something like:"
echo "    edit-image --config configs/json_file.json --verbose"