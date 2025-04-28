#!/usr/bin/env python3
# fine_tune.py - Скрипт для fine-tuning Mistral 7B на криптовалютных данных

import os
import torch
import logging
import json
import argparse
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    PeftModel
)
from trl import SFTTrainer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fine_tuning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CryptoFineTuner:
    """
    Класс для fine-tuning модели Mistral 7B на криптовалютных данных
    с использованием LoRA (Low-Rank Adaptation).
    """
    
    def __init__(
        self,
        base_model_name="mistralai/Mistral-7B-Instruct-v0.2",
        output_dir="./crypto_mistral_lora",
        dataset_path=None,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        learning_rate=2e-4,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_seq_length=512,
        load_in_8bit=True,
        cache_dir=None
    ):
        """
        Инициализация класса для fine-tuning.
        
        Args:
            base_model_name (str): Название базовой модели из Hugging Face Hub
            output_dir (str): Директория для сохранения результатов обучения
            dataset_path (str): Путь к датасету для fine-tuning
            lora_r (int): Ранг матрицы адаптации LoRA
            lora_alpha (int): Параметр масштабирования LoRA
            lora_dropout (float): Вероятность dropout в LoRA
            learning_rate (float): Скорость обучения
            num_train_epochs (int): Количество эпох обучения
            per_device_train_batch_size (int): Размер батча на устройство
            gradient_accumulation_steps (int): Шаги накопления градиента
            max_seq_length (int): Максимальная длина последовательности
            load_in_8bit (bool): Использовать ли 8-битную квантизацию
            cache_dir (str): Директория для кэширования модели
        """
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.dataset_path = dataset_path
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_seq_length = max_seq_length
        self.load_in_8bit = load_in_8bit
        self.cache_dir = cache_dir
        
        # Проверка доступности CUDA
        self.has_cuda = torch.cuda.is_available()
        if self.has_cuda:
            logger.info(f"CUDA доступна: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA недоступна, будет использоваться CPU")
            if load_in_8bit:
                logger.warning("8-битная квантизация недоступна на CPU, отключаем")
                self.load_in_8bit = False
        
        # Создание директории для результатов
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Инициализация токенизатора
        logger.info(f"Загрузка токенизатора {base_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            cache_dir=cache_dir
        )
        
        # Настройка токенизатора
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def prepare_dataset(self, dataset_path=None):
        """
        Подготовка датасета для fine-tuning.
        
        Args:
            dataset_path (str): Путь к датасету (JSON или CSV)
            
        Returns:
            Dataset: Подготовленный датасет
        """
        dataset_path = dataset_path or self.dataset_path
        
        if not dataset_path:
            raise ValueError("Необходимо указать путь к датасету")
        
        logger.info(f"Загрузка датасета из {dataset_path}...")
        
        # Определение формата файла
        file_extension = os.path.splitext(dataset_path)[1].lower()
        
        try:
            if file_extension == '.json':
                # Загрузка JSON датасета
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Преобразование в формат Dataset
                dataset = Dataset.from_dict({
                    'text': [self.format_instruction_sample(item) for item in data]
                })
                
            elif file_extension == '.csv':
                # Загрузка CSV датасета
                dataset = load_dataset('csv', data_files=dataset_path)['train']
                
                # Проверка наличия необходимых колонок
                required_columns = ['instruction', 'input', 'output']
                if not all(col in dataset.column_names for col in required_columns):
                    # Если нет нужных колонок, предполагаем, что есть колонка 'text'
                    if 'text' in dataset.column_names:
                        pass  # Используем как есть
                    else:
                        raise ValueError(f"CSV файл должен содержать колонки {required_columns} или 'text'")
                else:
                    # Форматирование датасета
                    dataset = dataset.map(
                        lambda x: {'text': self.format_instruction_sample(x)},
                        remove_columns=dataset.column_names
                    )
            else:
                raise ValueError(f"Неподдерживаемый формат файла: {file_extension}")
            
            logger.info(f"Датасет успешно загружен, {len(dataset)} примеров")
            return dataset
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке датасета: {str(e)}")
            raise
    
    def format_instruction_sample(self, item):
        """
        Форматирование примера в формат инструкции для Mistral.
        
        Args:
            item (dict): Пример данных с ключами 'instruction', 'input', 'output'
            
        Returns:
            str: Отформатированный текст
        """
        # Проверка формата данных
        if isinstance(item, dict):
            if 'instruction' in item and 'output' in item:
                instruction = item['instruction']
                input_text = item.get('input', '')
                output = item['output']
                
                # Форматирование в стиле Mistral Instruct
                if input_text:
                    formatted_text = f"<s>[INST] {instruction}\n\n{input_text} [/INST] {output}</s>"
                else:
                    formatted_text = f"<s>[INST] {instruction} [/INST] {output}</s>"
                
                return formatted_text
            elif 'text' in item:
                # Если уже есть готовый текст
                return item['text']
        
        # Если это просто строка
        if isinstance(item, str):
            return item
            
        # В случае неизвестного формата
        logger.warning(f"Неизвестный формат данных: {item}")
        return str(item)
    
    def load_model(self):
        """
        Загрузка базовой модели для fine-tuning.
        
        Returns:
            AutoModelForCausalLM: Загруженная модель
        """
        logger.info(f"Загрузка базовой модели {self.base_model_name}...")
        
        try:
            # Загрузка модели с квантизацией
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                load_in_8bit=self.load_in_8bit,
                device_map="auto",
                torch_dtype=torch.float16 if self.has_cuda else torch.float32,
                cache_dir=self.cache_dir
            )
            
            # Подготовка модели для обучения с квантизацией
            if self.load_in_8bit:
                model = prepare_model_for_kbit_training(model)
            
            logger.info("Модель успешно загружена")
            return model
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {str(e)}")
            raise
    
    def setup_lora(self, model):
        """
        Настройка LoRA для модели.
        
        Args:
            model: Базовая модель
            
        Returns:
            PeftModel: Модель с настроенным LoRA
        """
        logger.info("Настройка LoRA адаптера...")
        
        try:
            # Конфигурация LoRA
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]
            )
            
            # Применение LoRA к модели
            model = get_peft_model(model, lora_config)
            
            # Вывод информации о параметрах
            model.print_trainable_parameters()
            
            logger.info("LoRA адаптер успешно настроен")
            return model
            
        except Exception as e:
            logger.error(f"Ошибка при настройке LoRA: {str(e)}")
            raise
    
    def train(self, dataset=None):
        """
        Обучение модели на датасете.
        
        Args:
            dataset: Подготовленный датасет (если None, будет использован self.prepare_dataset())
            
        Returns:
            str: Путь к сохраненной модели
        """
        logger.info("Начало процесса fine-tuning...")
        
        try:
            # Подготовка датасета
            if dataset is None:
                dataset = self.prepare_dataset()
            
            # Загрузка и подготовка модели
            model = self.load_model()
            model = self.setup_lora(model)
            
            # Настройка аргументов обучения
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                per_device_train_batch_size=self.per_device_train_batch_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                learning_rate=self.learning_rate,
                num_train_epochs=self.num_train_epochs,
                weight_decay=0.01,
                save_strategy="epoch",
                save_total_limit=3,
                logging_steps=10,
                logging_dir=f"{self.output_dir}/logs",
                fp16=self.has_cuda,
                optim="adamw_torch",
                lr_scheduler_type="cosine",
                warmup_ratio=0.1,
                report_to="none"
            )
            
            # Настройка тренера
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_length,
                dataset_text_field="text"
            )
            
            # Запуск обучения
            logger.info("Запуск обучения...")
            trainer.train()
            
            # Сохранение модели
            final_model_path = f"{self.output_dir}/final"
            trainer.save_model(final_model_path)
            logger.info(f"Модель успешно обучена и сохранена в {final_model_path}")
            
            return final_model_path
            
        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {str(e)}")
            raise
    
    def create_sample_dataset(self, output_path="./crypto_sample_dataset.json", num_samples=10):
        """
        Создание примера датасета для fine-tuning.
        
        Args:
            output_path (str): Путь для сохранения примера датасета
            num_samples (int): Количество примеров
            
        Returns:
            str: Путь к созданному датасету
        """
        logger.info(f"Создание примера датасета с {num_samples} примерами...")
        
        # Примеры данных для fine-tuning
        samples = [
            {
                "instruction": "Проанализируй текущую ситуацию с Bitcoin",
                "input": "",
                "output": "Bitcoin в настоящее время торгуется на уровне $65,000. Технические индикаторы показывают бычий тренд с поддержкой на уровне $62,000. Объем торгов увеличился на 15% за последние 24 часа, что указывает на растущий интерес. Рекомендация: удерживать позиции с целевой ценой $70,000 в краткосрочной перспективе."
            },
            {
                "instruction": "Сравни Ethereum и Solana по скорости транзакций и комиссиям",
                "input": "",
                "output": "Ethereum обрабатывает около 15-30 транзакций в секунду со средней комиссией $5-20, в зависимости от загруженности сети. Solana значительно быстрее и дешевле, обрабатывая до 65,000 транзакций в секунду с комиссией менее $0.01. Однако Ethereum более децентрализован и имеет более широкую экосистему приложений и разработчиков."
            },
            {
                "instruction": "Объясни, что такое DeFi и какие основные протоколы существуют",
                "input": "",
                "output": "DeFi (Decentralized Finance) — это экосистема финансовых приложений, построенных на блокчейн-технологиях, которые стремятся воссоздать и улучшить традиционные финансовые услуги без централизованных посредников. Основные протоколы включают: Uniswap (децентрализованная биржа), Aave и Compound (кредитование/заимствование), MakerDAO (стабильные монеты), Curve (обмен стейблкоинов), Yearn Finance (автоматизированные инвестиционные стратегии) и Synthetix (синтетические активы)."
            },
            {
                "instruction": "Расскажи о технологии Proof of Stake и ее преимуществах перед Proof of Work",
                "input": "",
                "output": "Proof of Stake (PoS) — это механизм консенсуса, где валидаторы блоков выбираются на основе количества монет, которые они стейкают (блокируют) в сети. Преимущества перед Proof of Work (PoW): 1) Энергоэффективность — потребляет значительно меньше электроэнергии; 2) Масштабируемость — обрабатывает больше транзакций в секунду; 3) Безопасность — атака 51% требует владения большим количеством монет, что экономически невыгодно; 4) Доступность — не требует специализированного оборудования для майнинга; 5) Экологичность — меньший углеродный след."
            },
            {
                "instruction": "Объясни концепцию NFT и их применение в реальном мире",
                "input": "",
                "output": "NFT (Non-Fungible Token) — это уникальные цифровые активы, представляющие право собственности на конкретный предмет или контент в блокчейне. В отличие от криптовалют, каждый NFT уникален и не взаимозаменяем. Применения в реальном мире включают: 1) Цифровое искусство — художники продают оригинальные работы; 2) Коллекционные предметы — цифровые карточки, виртуальные питомцы; 3) Игровые активы — внутриигровые предметы, персонажи; 4) Недвижимость в метавселенных; 5) Билеты на мероприятия; 6) Сертификаты подлинности для физических товаров; 7) Документы о праве собственности; 8) Идентификация и доступ к эксклюзивным сообществам."
            }
        ]
        
        # Дополнение до нужного количества примеров
        while len(samples) < num_samples:
            samples.append(samples[len(samples) % 5])
        
        # Сохранение в файл
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples[:num_samples], f, ensure_ascii=False, indent=2)
        
        logger.info(f"Пример датасета создан и сохранен в {output_path}")
        return output_path
    
    @staticmethod
    def merge_lora_weights(base_model_path, lora_model_path, output_path):
        """
        Объединение весов LoRA с базовой моделью.
        
        Args:
            base_model_path (str): Путь к базовой модели
            lora_model_path (str): Путь к LoRA адаптеру
            output_path (str): Путь для сохранения объединенной модели
            
        Returns:
            str: Путь к объединенной модели
        """
        logger.info(f"Объединение базовой модели с LoRA адаптером...")
        
        try:
            # Загрузка базовой модели
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Загрузка токенизатора
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            
            # Загрузка модели с LoRA адаптером
            model = PeftModel.from_pretrained(base_model, lora_model_path)
            
            # Объединение весов
            model = model.merge_and_unload()
            
            # Сохранение объединенной модели
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            logger.info(f"Модель успешно объединена и сохранена в {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Ошибка при объединении моделей: {str(e)}")
            raise


# Функция для запуска из командной строки
def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Mistral 7B на криптовалютных данных")
    
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Название базовой модели из Hugging Face Hub")
    parser.add_argument("--output_dir", type=str, default="./crypto_mistral_lora",
                        help="Директория для сохранения результатов обучения")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Путь к датасету для fine-tuning")
    parser.add_argument("--create_sample", action="store_true",
                        help="Создать пример датасета")
    parser.add_argument("--sample_path", type=str, default="./crypto_sample_dataset.json",
                        help="Путь для сохранения примера датасета")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Количество примеров в примере датасета")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="Ранг матрицы адаптации LoRA")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="Параметр масштабирования LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Вероятность dropout в LoRA")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Скорость обучения")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Количество эпох обучения")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Размер батча на устройство")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Шаги накопления градиента")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Максимальная длина последовательности")
    parser.add_argument("--no_8bit", action="store_true",
                        help="Отключить 8-битную квантизацию")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Директория для кэширования модели")
    
    args = parser.parse_args()
    
    # Создание экземпляра класса
    fine_tuner = CryptoFineTuner(
        base_model_name=args.base_model,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        load_in_8bit=not args.no_8bit,
        cache_dir=args.cache_dir
    )
    
    # Создание примера датасета, если нужно
    if args.create_sample:
        sample_path = fine_tuner.create_sample_dataset(
            output_path=args.sample_path,
            num_samples=args.num_samples
        )
        print(f"Пример датасета создан: {sample_path}")
        
        # Если путь к датасету не указан, используем созданный пример
        if args.dataset_path is None:
            args.dataset_path = sample_path
            fine_tuner.dataset_path = sample_path
    
    # Запуск обучения, если указан путь к датасету
    if args.dataset_path:
        final_model_path = fine_tuner.train()
        print(f"Обучение завершено. Модель сохранена в: {final_model_path}")
    else:
        print("Путь к датасету не указан. Используйте --dataset_path или --create_sample")


if __name__ == "__main__":
    main()
