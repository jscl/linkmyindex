#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if we're already in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    # Try to activate .venv if it exists
    if [ -d "$SCRIPT_DIR/.venv" ]; then
        echo "Activating virtual environment..."
        source "$SCRIPT_DIR/.venv/bin/activate"
    else
        echo "Warning: No virtual environment found at $SCRIPT_DIR/.venv"
        echo "Running with system Python. Consider running 'uv sync' or creating a venv."
    fi
fi

# Run the linkmyindex script with all passed arguments
python "$SCRIPT_DIR/src/linkmyindex.py" "$@"
