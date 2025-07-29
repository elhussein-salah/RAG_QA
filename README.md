# RAG Local LLM - Arabic Question Answering System

A comprehensive Retrieval-Augmented Generation (RAG) pipeline for answering questions in Arabic using vector embeddings and local LLM (Ollama with gemma3:1b model).

## 🌟 Features

- **Arabic Language Support**: Built specifically for Arabic text processing and understanding
- **Vector-based Retrieval**: Uses FAISS index with Arabic BERT embeddings for efficient context retrieval
- **Local LLM Integration**: Powered by Ollama with gemma3:1b model for privacy and offline usage
- **Modular Architecture**: Well-structured codebase with separate modules for different functionalities
- **Multiple Interfaces**: Both web-based (Gradio) and command-line interfaces
- **Sample Dataset**: Includes Arabic context-question-answer pairs for immediate testing

## 🏗️ Project Structure

```
rag_local_llm/
├── data_preparation.py     # Dataset loading and preprocessing
├── embedding.py           # Arabic BERT embeddings and FAISS index
├── retrieval.py          # Context retrieval functionality
├── llm_generation.py     # Ollama LLM integration
├── rag_pipeline.py       # Main RAG pipeline orchestration
├── app.py               # Gradio web interface + CLI
├── requirements.txt     # Python dependencies
├── setup.sh            # Setup script for Unix/Linux/macOS
├── setup.bat           # Setup script for Windows
└── README.md           # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git (for cloning)
- Internet connection (for initial model downloads)

### Installation

#### For Windows:
```bash
# Clone the repository
git clone <repository-url>
cd rag_local_llm

# Run setup script
setup.bat
```

#### For Linux/macOS:
```bash
# Clone the repository
git clone <repository-url>
cd rag_local_llm

# Make setup script executable and run
chmod +x setup.sh
./setup.sh
```

#### Manual Installation:

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install Ollama:**
   - Visit [https://ollama.ai/download](https://ollama.ai/download)
   - Download and install for your platform

3. **Pull the required model:**
```bash
ollama pull gemma3:1b
```

## 🎯 Usage

### Web Interface (Recommended)

Launch the Gradio web interface:

```bash
python app.py
```

Then open your browser to `http://localhost:7860`

Features:
- User-friendly Arabic interface
- Real-time question answering
- Context visualization
- System status monitoring
- Sample questions for testing

### Command Line Interface

For terminal-based interaction:

```bash
python app.py --cli
```

### Direct Pipeline Usage

For programmatic usage:

```python
from rag_pipeline import RAGPipeline

# Initialize and setup
rag = RAGPipeline()
rag.initialize()

# Ask a question
result = rag.answer_question("ما هو الذكاء الاصطناعي؟")
print(result['answer'])
```

## 📊 Sample Dataset

The project includes a sample Arabic dataset with 6 topics:

1. **Artificial Intelligence** - تعريف الذكاء الاصطناعي
2. **Egypt** - معلومات عن جمهورية مصر العربية  
3. **Machine Learning** - مفهوم التعلم الآلي
4. **Great Pyramid** - معلومات عن الهرم الأكبر
5. **Arabic Language** - حقائق عن اللغة العربية
6. **Egypt Vision 2030** - رؤية مصر 2030

## 🔧 Configuration

### Embedding Model
- Default: `aubmindlab/bert-base-arabertv2`
- Can be changed in `RAGPipeline()` initialization

### LLM Model
- Default: `gemma3:1b`
- Can be changed to other Ollama models

### Retrieval Parameters
- `top_k`: Number of contexts to retrieve (default: 3)
- `temperature`: LLM creativity level (default: 0.7)

## 📁 File Descriptions

### Core Modules

- **`data_preparation.py`**: Dataset loading, preprocessing, and sample data creation
- **`embedding.py`**: Arabic BERT model integration and FAISS index management
- **`retrieval.py`**: Context retrieval using vector similarity search
- **`llm_generation.py`**: Ollama integration for answer generation
- **`rag_pipeline.py`**: Main orchestrator combining all components

### Applications

- **`app.py`**: Complete application with web and CLI interfaces

### Setup

- **`setup.sh`** / **`setup.bat`**: Automated setup scripts
- **`requirements.txt`**: Python dependencies list

## 🛠️ Technical Details

### Architecture

1. **Data Preparation**: Load and preprocess Arabic text data
2. **Embedding Generation**: Create vector embeddings using Arabic BERT
3. **Index Building**: Store embeddings in FAISS index for fast retrieval
4. **Question Processing**: Convert user questions to embeddings
5. **Context Retrieval**: Find most similar contexts using cosine similarity
6. **Answer Generation**: Use Ollama LLM to generate contextual answers

### Models Used

- **Embedding Model**: `aubmindlab/bert-base-arabertv2`
  - Specialized Arabic BERT model
  - 768-dimensional embeddings
  - Optimized for Arabic text understanding

- **LLM Model**: `gemma3:1b`
  - Lightweight but capable model
  - Runs efficiently on consumer hardware
  - Good Arabic language support

### Performance Considerations

- **First Run**: Downloading models may take 10-15 minutes
- **Embedding Creation**: ~30 seconds for sample dataset
- **Query Response**: ~2-5 seconds per question
- **Memory Usage**: ~2-4 GB RAM (depending on model size)

## 🔍 Testing

Test individual components:

```bash
# Test data preparation
python data_preparation.py

# Test embedding creation
python embedding.py

# Test retrieval
python retrieval.py

# Test LLM generation
python llm_generation.py

# Test full pipeline
python rag_pipeline.py
```

## 🚨 Troubleshooting

### Common Issues

1. **Ollama not found**:
   - Ensure Ollama is installed and in PATH
   - Try restarting your terminal/command prompt

2. **Model download fails**:
   - Check internet connection
   - Try: `ollama pull gemma3:1b` manually

3. **CUDA/GPU issues**:
   - The system automatically falls back to CPU
   - For GPU support, install appropriate PyTorch version

4. **Arabic text rendering**:
   - Ensure your terminal/browser supports Arabic text
   - Use the web interface for better Arabic support

### Performance Tips

- **GPU Acceleration**: Install CUDA-compatible PyTorch for faster embeddings
- **Model Selection**: Use larger models for better quality (e.g., `gemma3:8b`)
- **Batch Processing**: Process multiple questions using `batch_answer_questions()`

## 🛡️ Privacy & Security

- **Local Processing**: All computation happens locally
- **No Data Transmission**: No external API calls after initial setup
- **Data Control**: Complete control over your data and models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For the transformers library and Arabic BERT model
- **Facebook AI**: For FAISS vector database
- **Ollama**: For local LLM infrastructure
- **Gradio**: For the web interface framework
- **AubmindLab**: For the Arabic BERT model

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the console/terminal output for error messages
3. Ensure all dependencies are properly installed
4. Verify Ollama is running: `ollama list`

---

**Happy questioning! 🎉 استمتع بطرح الأسئلة!**
