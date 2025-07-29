@echo off
REM RAG Local LLM Setup Script for Windows
REM This script sets up the environment and installs Ollama

echo 🚀 Setting up RAG Local LLM Project
echo ==================================

REM Check if Python is installed
echo 🐍 Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
) else (
    for /f "tokens=2" %%i in ('python --version') do echo ✅ Python %%i found
)

REM Check if pip is installed
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip is not installed
    echo Please reinstall Python with pip included
    pause
    exit /b 1
) else (
    echo ✅ pip found
)

REM Install Python requirements
echo 📦 Installing Python requirements...
if exist requirements.txt (
    pip install -r requirements.txt
    if %errorlevel% equ 0 (
        echo ✅ Python packages installed successfully
    ) else (
        echo ❌ Failed to install Python packages
        pause
        exit /b 1
    )
) else (
    echo ❌ requirements.txt not found
    pause
    exit /b 1
)

REM Check if Ollama is installed
echo 🤖 Checking Ollama installation...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Ollama is not installed
    echo Please download and install Ollama from: https://ollama.ai/download
    echo After installation, restart this script
    pause
    exit /b 1
) else (
    echo ✅ Ollama found
)

REM Start Ollama service (it usually starts automatically on Windows)
echo 🔄 Ensuring Ollama service is running...
timeout /t 3 /nobreak >nul

REM Pull the required model
echo 📥 Pulling gemma3:1b model...
ollama pull gemma3:1b

if %errorlevel% equ 0 (
    echo ✅ gemma3:1b model pulled successfully
) else (
    echo ❌ Failed to pull gemma3:1b model
    echo Please make sure Ollama is running and try: ollama pull gemma3:1b
)

REM Test Ollama installation
echo 🧪 Testing Ollama installation...
ollama list

REM Create embeddings directory
echo 📁 Creating embeddings directory...
if not exist embeddings mkdir embeddings

echo.
echo 🎉 Setup completed!
echo.
echo To run the application:
echo   • For Gradio web interface: python app.py
echo   • For command-line interface: python app.py --cli
echo   • For pipeline testing: python rag_pipeline.py
echo.
echo 📝 Note: On first run, the system will download the Arabic BERT model
echo    and create embeddings. This may take a few minutes.
echo.
pause
