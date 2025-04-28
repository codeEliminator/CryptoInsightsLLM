#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import logging
import time
import os
import json
from datetime import datetime

# Импорт нашего класса CryptoLLM
from crypto_llm import CryptoLLM

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация FastAPI приложения
app = FastAPI(
    title="CryptoAI LLM API",
    description="API для локальной LLM на базе Mistral 7B, специализированной на анализе криптовалют",
    version="1.0.0"
)

# Настройка CORS для доступа из React Native приложения
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модели данных для API
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Текст запроса к модели")
    max_new_tokens: int = Field(512, description="Максимальное количество новых токенов")
    temperature: float = Field(0.7, description="Температура для семплирования (0.0-1.0)")
    top_p: float = Field(0.9, description="Параметр nucleus sampling (0.0-1.0)")
    top_k: int = Field(50, description="Количество токенов для top-k sampling")
    repetition_penalty: float = Field(1.1, description="Штраф за повторение")
    do_sample: bool = Field(True, description="Использовать ли семплирование")
    system_prompt: Optional[str] = Field(None, description="Системный промпт для инструктирования модели")

class GenerateResponse(BaseModel):
    text: str = Field(..., description="Сгенерированный текст")
    generation_time: float = Field(..., description="Время генерации в секундах")
    prompt_tokens: int = Field(..., description="Количество токенов в запросе")
    completion_tokens: int = Field(..., description="Количество токенов в ответе")
    total_tokens: int = Field(..., description="Общее количество токенов")

class CryptoAnalysisRequest(BaseModel):
    crypto_name: str = Field(..., description="Название криптовалюты (например, 'Bitcoin', 'Ethereum')")
    analysis_type: str = Field("general", description="Тип анализа: 'general', 'technical', 'sentiment', 'prediction'")

class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., description="Название модели")
    device: str = Field(..., description="Устройство для вычислений (CPU/GPU)")
    quantization: str = Field(..., description="Тип квантизации")
    model_size: float = Field(..., description="Размер модели в миллионах параметров")
    cuda_available: bool = Field(..., description="Доступность CUDA")
    cuda_device: Optional[str] = Field(None, description="Название CUDA устройства")
    api_version: str = Field(..., description="Версия API")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Статус сервера")
    timestamp: str = Field(..., description="Текущее время")
    uptime: float = Field(..., description="Время работы сервера в секундах")

# Глобальные переменные
llm = None
start_time = time.time()
request_history = []
MAX_HISTORY_SIZE = 100  # Максимальное количество запросов в истории

# Функция для ленивой инициализации модели
def get_llm():
    global llm
    if llm is None:
        logger.info("Инициализация модели CryptoLLM...")
        try:
            # Загрузка модели при первом запросе
            llm = CryptoLLM(
                model_name="mistralai/Mistral-7B-Instruct-v0.2",
                device="auto",
                load_in_8bit=True,
                cache_dir="./model_cache"
            )
            logger.info("Модель успешно инициализирована")
        except Exception as e:
            logger.error(f"Ошибка при инициализации модели: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Ошибка инициализации модели: {str(e)}")
    return llm

# Функция для логирования запросов
def log_request(request_type: str, request_data: Dict[str, Any], response_data: Dict[str, Any], duration: float):
    global request_history
    
    # Создаем запись о запросе
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "request_type": request_type,
        "request_data": request_data,
        "response_summary": {
            "generation_time": duration,
            "tokens": response_data.get("total_tokens", 0)
        }
    }
    
    # Добавляем в историю и ограничиваем размер
    request_history.append(log_entry)
    if len(request_history) > MAX_HISTORY_SIZE:
        request_history = request_history[-MAX_HISTORY_SIZE:]
    
    # Сохраняем историю в файл (асинхронно)
    try:
        with open("request_history.json", "w") as f:
            json.dump(request_history, f, indent=2)
    except Exception as e:
        logger.warning(f"Не удалось сохранить историю запросов: {str(e)}")

# Маршруты API
@app.get("/", tags=["Информация"])
async def root():
    """
    Корневой маршрут с информацией об API.
    """
    return {
        "name": "CryptoAI LLM API",
        "version": "0.1.0",
        "description": "API для локальной LLM на базе Mistral 7B, специализированной на анализе криптовалют",
        "endpoints": {
            "/generate": "Генерация текста на основе промпта",
            "/analyze": "Анализ криптовалюты",
            "/model-info": "Информация о модели",
            "/health": "Проверка состояния сервера"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Информация"])
async def health_check():
    """
    Проверка состояния сервера.
    """
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "uptime": time.time() - start_time
    }

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Информация"])
async def model_info(llm: CryptoLLM = Depends(get_llm)):
    """
    Получение информации о модели.
    """
    info = llm.get_model_info()
    info["api_version"] = "1.0.0"
    return info

@app.post("/generate", response_model=GenerateResponse, tags=["Генерация"])
async def generate_text(
    request: GenerateRequest, 
    background_tasks: BackgroundTasks,
    llm: CryptoLLM = Depends(get_llm)
):
    """
    Генерация текста на основе промпта.
    """
    logger.info(f"Запрос на генерацию текста: {request.prompt[:50]}...")
    
    start = time.time()
    try:
        # Генерация ответа
        response = llm.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
            system_prompt=request.system_prompt
        )
        
        # Подсчет токенов
        prompt_tokens = len(llm.tokenizer.encode(request.prompt))
        completion_tokens = len(llm.tokenizer.encode(response))
        total_tokens = prompt_tokens + completion_tokens
        
        # Формирование ответа
        generation_time = time.time() - start
        result = {
            "text": response,
            "generation_time": generation_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        
        # Логирование запроса в фоновом режиме
        background_tasks.add_task(
            log_request, 
            "generate", 
            request.dict(), 
            result, 
            generation_time
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка при генерации текста: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {str(e)}")

@app.post("/analyze", response_model=GenerateResponse, tags=["Анализ"])
async def analyze_crypto(
    request: CryptoAnalysisRequest, 
    background_tasks: BackgroundTasks,
    llm: CryptoLLM = Depends(get_llm)
):
    """
    Анализ криптовалюты.
    """
    logger.info(f"Запрос на анализ криптовалюты: {request.crypto_name}, тип: {request.analysis_type}")
    
    start = time.time()
    try:
        # Анализ криптовалюты
        response = llm.analyze_crypto(
            crypto_name=request.crypto_name,
            analysis_type=request.analysis_type
        )
        
        # Формирование промпта для логирования
        if request.analysis_type == "general":
            prompt = f"Предоставь общий анализ текущего состояния {request.crypto_name}."
        elif request.analysis_type == "technical":
            prompt = f"Проведи технический анализ {request.crypto_name}."
        elif request.analysis_type == "sentiment":
            prompt = f"Проанализируй настроение рынка относительно {request.crypto_name}."
        elif request.analysis_type == "prediction":
            prompt = f"Предоставь обоснованный прогноз для {request.crypto_name}."
        else:
            prompt = f"Проанализируй {request.crypto_name}."
        
        # Подсчет токенов
        prompt_tokens = len(llm.tokenizer.encode(prompt))
        completion_tokens = len(llm.tokenizer.encode(response))
        total_tokens = prompt_tokens + completion_tokens
        
        # Формирование ответа
        generation_time = time.time() - start
        result = {
            "text": response,
            "generation_time": generation_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        
        # Логирование запроса в фоновом режиме
        background_tasks.add_task(
            log_request, 
            "analyze", 
            request.dict(), 
            result, 
            generation_time
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка при анализе криптовалюты: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")

if __name__ == "__main__":
    os.makedirs("./model_cache", exist_ok=True)
    
    # Запускаем сервер
    logger.info("Запуск CryptoAI LLM API сервера...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
