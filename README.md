# CryptoAI Insights - Local LLM (Mistral 7B)

## Overview

CryptoAI Insights provides a **fully local** Large Language Model (LLM) solution based on **Mistral 7B**, optimized for cryptocurrency market analysis. It features:

- **CryptoLLM**: A core class to interact with the model.
- **FastAPI server**: For API interaction.
- **CryptoFineTuner**: Fine-tuning on crypto-specific datasets.
- **React Native integration**: To seamlessly integrate AI capabilities into mobile apps.

This system allows running, fine-tuning, and querying a crypto-specialized LLM without relying on external services.

## Why Mistral 7B?

- **Efficient performance vs. resource use**
- **Outperforms LLaMA 2 13B** on benchmarks
- **Fine-tuning support with strong tools**
- **Apache 2.0 License**: Allows commercial use
- **Multilingual capabilities**
- **Active community support**

## Architecture

```
React Native App ⇄ FastAPI Server ⇄ Mistral 7B LLM
                                    ▲
                                    │
                          Fine-tuning Pipeline
                                    ▲
                                    │
                          Crypto Data Sources
```

## Project Structure

```
crypto_llm/
├── crypto_llm.py         # Core LLM interface
├── app.py                # FastAPI server
├── fine_tune.py          # Fine-tuning tool
├── model_cache/          # Model cache directory
├── crypto_mistral_lora/  # Fine-tuned model output
└── datasets/             # Crypto datasets
```

---

## System Requirements

### Minimum:
- CPU: 4+ cores
- RAM: 16 GB
- Storage: 20 GB
- GPU: NVIDIA with 8+ GB VRAM

### Recommended:
- CPU: 8+ cores
- RAM: 32 GB
- Storage: 50 GB
- GPU: NVIDIA with 16+ GB VRAM

---

## Installation

```bash
# Create and activate a virtual environment
python -m venv crypto_llm_env
source crypto_llm_env/bin/activate  # Linux/Mac
crypto_llm_env\Scripts\activate     # Windows

# Install core dependencies
pip install torch transformers accelerate bitsandbytes peft trl datasets fastapi uvicorn sentencepiece

# Optional (GPU support)
pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12
```

---

## Using CryptoLLM

### Initialization
```python
from crypto_llm import CryptoLLM

llm = CryptoLLM(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    device="cuda",  # or "cpu", or "auto"
    load_in_8bit=True,
    cache_dir="./model_cache"
)
```

### Generate Text
```python
response = llm.generate("Explain Bitcoin.")
print(response)
```

### Crypto Analysis
```python
analysis = llm.analyze_crypto("Bitcoin", "general")
print(analysis)
```

### Model Information
```python
info = llm.get_model_info()
print(info)
```

---

## Running the FastAPI Server

```bash
python app.py
# or using uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Available Endpoints
- `GET /` — API Info
- `GET /health` — Health Check
- `GET /model-info` — Model Information
- `POST /generate` — Text Generation
- `POST /analyze` — Crypto Analysis

### API Documentation
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Example Requests
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What is DeFi?", "max_new_tokens": 512}'
```

---

## Fine-tuning on Crypto Data

### Dataset Format
```json
[
  {"instruction": "Analyze Bitcoin.", "input": "", "output": "Bitcoin trades at $65,000 with bullish indicators..."},
  {"instruction": "Compare Ethereum and Solana.", "input": "", "output": "Ethereum handles 15-30 TPS, Solana up to 65,000 TPS..."}
]
```

### Creating Sample Dataset
```bash
python fine_tune.py --create_sample --sample_path ./datasets/crypto_sample.json --num_samples 20
```

### Fine-tuning
```bash
python fine_tune.py --dataset_path ./datasets/crypto_sample.json --output_dir ./crypto_mistral_lora
```

---

## React Native Integration

### API Client Example (`api.js`)
```javascript
const API_BASE_URL = 'http://your-api-server:8000';

export const generateAIResponse = async (prompt, options = {}) => {
  const response = await fetch(`${API_BASE_URL}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, max_new_tokens: options.maxLength || 512, temperature: options.temperature || 0.7 })
  });
  const data = await response.json();
  return data.text;
};
```

---

## Performance Optimization

- **Quantization**: Load models in 8-bit by default for lower memory usage.
- **Caching**: Use `cache_dir` for model downloads.
- **FastAPI Optimizations**:
  ```bash
  gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
  ```
- **Prompt Engineering**: Use lower `max_new_tokens`, temperature tuning, and repetition penalties.

---

## Troubleshooting

### CUDA Errors
- Reduce model/batch size
- Free GPU memory
- Use CPU mode if necessary

### Model Loading Issues
- Check internet connection
- Check storage space
- Download models manually if needed

### API Connection Issues
- Ensure server is running
- Validate IP and port
- Configure CORS if accessing from browsers

---

## Development Roadmap

### Short-Term (1-2 weeks)
- Basic model + API + RN integration
- Initial fine-tuning

### Mid-Term (1-2 months)
- Fine-tuning on larger datasets
- Performance optimization
- Specialized crypto features

### Long-Term (3+ months)
- Continuous learning
- Real-time data integration
- Specialized sub-models for different crypto tasks

---

## Conclusion

CryptoAI Insights empowers your crypto app with a fast, powerful, fully local LLM based on **Mistral 7B** — ideal for independent analysis, predictions, and market understanding, all with easy mobile integration.
