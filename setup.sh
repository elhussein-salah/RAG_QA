#!/bin/bash

# RAG Local LLM Setup Script
# This script sets up Ollama and pulls the required model

echo "🚀 Setting up RAG Local LLM Project"
echo "=================================="

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Ollama on different platforms
install_ollama() {
    echo "📥 Installing Ollama..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl -fsSL https://ollama.ai/install.sh | sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command_exists brew; then
            brew install ollama
        else
            echo "Please install Homebrew first, then run: brew install ollama"
            echo "Or download from: https://ollama.ai/download"
            exit 1
        fi
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows (Git Bash/MSYS2)
        echo "Please download and install Ollama from: https://ollama.ai/download"
        echo "After installation, run this script again."
        exit 1
    else
        echo "Unsupported operating system. Please install Ollama manually from: https://ollama.ai/download"
        exit 1
    fi
}

# Check if Python is installed
echo "🐍 Checking Python installation..."
if ! command_exists python3; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
else
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo "✅ Python $python_version found"
fi

# Check if pip is installed
if ! command_exists pip3; then
    echo "❌ pip3 is not installed. Please install pip3."
    exit 1
else
    echo "✅ pip3 found"
fi

# Install Python requirements
echo "📦 Installing Python requirements..."
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo "✅ Python packages installed successfully"
    else
        echo "❌ Failed to install Python packages"
        exit 1
    fi
else
    echo "❌ requirements.txt not found"
    exit 1
fi

# Check if Ollama is installed
echo "🤖 Checking Ollama installation..."
if ! command_exists ollama; then
    echo "Ollama is not installed. Installing..."
    install_ollama
else
    echo "✅ Ollama found"
fi

# Start Ollama service (if not already running)
echo "🔄 Starting Ollama service..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux - start as systemd service
    if command_exists systemctl; then
        sudo systemctl start ollama
        sudo systemctl enable ollama
    else
        # Start manually in background
        nohup ollama serve > ollama.log 2>&1 &
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - start as background service
    if command_exists brew; then
        brew services start ollama
    else
        nohup ollama serve > ollama.log 2>&1 &
    fi
fi

# Wait a moment for Ollama to start
sleep 3

# Pull the required model
echo "📥 Pulling gemma3:1b model..."
ollama pull gemma3:1b

if [ $? -eq 0 ]; then
    echo "✅ gemma3:1b model pulled successfully"
else
    echo "❌ Failed to pull gemma3:1b model"
    echo "Please make sure Ollama is running and try: ollama pull gemma3:1b"
fi

# Test Ollama installation
echo "🧪 Testing Ollama installation..."
ollama list

# Create embeddings directory
echo "📁 Creating embeddings directory..."
mkdir -p embeddings

echo ""
echo "🎉 Setup completed!"
echo ""
echo "To run the application:"
echo "  • For Gradio web interface: python3 app.py"
echo "  • For command-line interface: python3 app.py --cli"
echo "  • For pipeline testing: python3 rag_pipeline.py"
echo ""
echo "📝 Note: On first run, the system will download the Arabic BERT model"
echo "   and create embeddings. This may take a few minutes."
echo ""
