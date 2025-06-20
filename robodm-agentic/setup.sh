#!/bin/bash

# Setup script for RoboDM Agentic Framework

echo "Setting up RoboDM Agentic Framework..."

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found. Run this script from the robodm-agentic directory."
    exit 1
fi

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Using virtual environment: $VIRTUAL_ENV"
    PYTHON_CMD="python"
    PIP_CMD="pip"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
else
    PYTHON_CMD="python"
    PIP_CMD="pip"
fi

echo "Using Python: $(which $PYTHON_CMD)"
echo "Using pip: $(which $PIP_CMD)"

# Install Python dependencies
echo "Installing Python dependencies..."
$PIP_CMD install -r requirements.txt

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
    
    if [ $? -eq 0 ]; then
        echo "Ollama installed successfully."
    else
        echo "Failed to install Ollama. Please install manually from https://ollama.ai"
    fi
else
    echo "Ollama is already installed."
fi

# Pull recommended models
echo "Pulling recommended models..."
echo "This may take a while for the first time..."

# Pull LLM for code generation
echo "Pulling Qwen 2.5 7B for code generation..."
ollama pull qwen2.5:7b

# Pull VLM for vision analysis
echo "Pulling LLaVA 7B for vision analysis..."
ollama pull llava:7b

# Verify installation
echo "Verifying installation..."

# Check if models are available
if ollama list | grep -q "qwen2.5:7b"; then
    echo "✓ Qwen 2.5 7B installed"
else
    echo "✗ Qwen 2.5 7B not found"
fi

if ollama list | grep -q "llava:7b"; then
    echo "✓ LLaVA 7B installed"
else
    echo "✗ LLaVA 7B not found"
fi

# Check Python dependencies
echo "Checking Python dependencies..."
$PYTHON_CMD -c "
try:
    import numpy
    print('✓ NumPy available')
except ImportError:
    print('✗ NumPy not available')

try:
    from PIL import Image
    print('✓ Pillow available')
except ImportError:
    print('✗ Pillow not available')

try:
    import ollama
    print('✓ Ollama Python client available')
except ImportError:
    print('✗ Ollama Python client not available')

# Check if we can import the main robodm package
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath('.'))))
try:
    import robodm
    print('✓ RoboDM package available')
except ImportError as e:
    print(f'✗ RoboDM package not available: {e}')
"

echo ""
echo "Setup complete!"
echo ""
echo "To test the installation, run:"
echo "  $PYTHON_CMD example_usage.py"
echo ""
echo "To ingest Open-X Embodiment data, run:"
echo "  $PYTHON_CMD ingest_oxe_demo.py"
echo ""
echo "Make sure to update the data paths in the example scripts to point to your trajectory data."
echo ""
echo "If you're using a virtual environment, make sure it's activated before running the scripts."
