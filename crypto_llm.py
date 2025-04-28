#!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CryptoLLM:
    """
    Класс для работы с локальной LLM на базе Mistral 7B,
    оптимизированной для анализа криптовалют.
    """
    
    def __init__(
        self, 
        model_name="mistralai/Mistral-7B-Instruct-v0.2", 
        device="auto",
        load_in_8bit=True,
        cache_dir=None
    ):
        """
        Инициализация модели Mistral 7B.
        
        Args:
            model_name (str): Название модели из Hugging Face Hub
            device (str): Устройство для вычислений ('cpu', 'cuda', 'auto')
            load_in_8bit (bool): Использовать ли 8-битную квантизацию
            cache_dir (str): Директория для кэширования модели
        """
        self.model_name = model_name
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.cache_dir = cache_dir
        
        logger.info(f"Инициализация CryptoLLM с моделью {model_name}")
        
        # Проверка доступности CUDA
        self.has_cuda = torch.cuda.is_available()
        if self.has_cuda:
            logger.info(f"CUDA доступна: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA недоступна, будет использоваться CPU")
            if load_in_8bit:
                logger.warning("8-битная квантизация недоступна на CPU, отключаем")
                self.load_in_8bit = False
        
        # Загрузка токенизатора
        logger.info("Загрузка токенизатора...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Настройка квантизации
        quantization_config = None
        if self.load_in_8bit:
            logger.info("Настройка 8-битной квантизации...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        
        # Загрузка модели
        logger.info("Загрузка модели (это может занять некоторое время)...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if self.has_cuda else torch.float32,
                cache_dir=cache_dir
            )
            logger.info("Модель успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {str(e)}")
            raise
    
    def generate(
        self, 
        prompt, 
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        do_sample=True,
        system_prompt=None
    ):
        """
        Генерация ответа на основе промпта.
        
        Args:
            prompt (str): Текст запроса
            max_new_tokens (int): Максимальное количество новых токенов
            temperature (float): Температура для семплирования
            top_p (float): Параметр nucleus sampling
            top_k (int): Количество токенов для top-k sampling
            repetition_penalty (float): Штраф за повторение
            do_sample (bool): Использовать ли семплирование
            system_prompt (str): Системный промпт для инструктирования модели
            
        Returns:
            str: Сгенерированный ответ
        """
        logger.info(f"Генерация ответа на запрос: {prompt[:50]}...")
        
        # Форматирование промпта в соответствии с форматом Mistral Instruct
        if system_prompt:
            formatted_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
        else:
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Токенизация входного текста
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        
        # Перемещение тензоров на нужное устройство
        if self.has_cuda and self.device != "cpu":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Генерация ответа
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Декодирование ответа
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлечение только ответа модели (без промпта)
            response = full_response.split("[/INST]")[-1].strip()
            
            logger.info(f"Ответ успешно сгенерирован ({len(response)} символов)")
            return response
            
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {str(e)}")
            raise
    
    def analyze_crypto(self, crypto_name, analysis_type="general"):
        """
        Специализированный метод для анализа криптовалюты.
        
        Args:
            crypto_name (str): Название криптовалюты (например, "Bitcoin", "Ethereum")
            analysis_type (str): Тип анализа ("general", "technical", "sentiment", "prediction")
            
        Returns:
            str: Аналитический ответ о криптовалюте
        """
        logger.info(f"Анализ криптовалюты {crypto_name}, тип анализа: {analysis_type}")
        
        # Системный промпт для криптоанализа
        system_prompt = """Ты — эксперт по криптовалютам и финансовому анализу. 
        Твоя задача — предоставить точный, информативный и объективный анализ криптовалют.
        Основывай свои ответы на фактах, технических индикаторах и рыночных тенденциях.
        Избегай спекуляций и необоснованных прогнозов."""
        
        # Формирование промпта в зависимости от типа анализа
        if analysis_type == "general":
            prompt = f"Предоставь общий анализ текущего состояния {crypto_name}. Включи информацию о цене, рыночной капитализации, объеме торгов и основных новостях."
        
        elif analysis_type == "technical":
            prompt = f"Проведи технический анализ {crypto_name}. Рассмотри ключевые уровни поддержки и сопротивления, индикаторы тренда (MA, MACD, RSI) и возможные паттерны графика."
        
        elif analysis_type == "sentiment":
            prompt = f"Проанализируй настроение рынка относительно {crypto_name}. Оцени общий сентимент инвесторов, активность в социальных медиа и влияние недавних новостей."
        
        elif analysis_type == "prediction":
            prompt = f"На основе текущих данных, предоставь обоснованный прогноз для {crypto_name} на ближайшую перспективу. Укажи возможные сценарии развития и факторы, которые могут повлиять на цену."
        
        else:
            prompt = f"Проанализируй {crypto_name} и предоставь полезную информацию для инвесторов."
        
        # Генерация ответа с использованием специализированного системного промпта
        return self.generate(prompt, system_prompt=system_prompt, temperature=0.3)
    
    def get_model_info(self):
        """
        Получение информации о модели.
        
        Returns:
            dict: Информация о модели
        """
        return {
            "model_name": self.model_name,
            "device": self.device if self.device != "auto" else ("cuda" if self.has_cuda else "cpu"),
            "quantization": "8-bit" if self.load_in_8bit else "None",
            "model_size": sum(p.numel() for p in self.model.parameters()) / 1_000_000,  # в миллионах параметров
            "cuda_available": self.has_cuda,
            "cuda_device": torch.cuda.get_device_name(0) if self.has_cuda else None,
        }


# Пример использования
if __name__ == "__main__":
    # Проверка работоспособности
    try:
        print("Инициализация CryptoLLM...")
        llm = CryptoLLM()
        
        print("\nИнформация о модели:")
        model_info = llm.get_model_info()
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        print("\nТестовая генерация:")
        response = llm.generate("Что такое Bitcoin?")
        print(f"Ответ: {response}")
        
        print("\nАнализ криптовалюты:")
        analysis = llm.analyze_crypto("Ethereum", "technical")
        print(f"Анализ: {analysis}")
        
    except Exception as e:
        print(f"Ошибка при тестировании: {str(e)}")
