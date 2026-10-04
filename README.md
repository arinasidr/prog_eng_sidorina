## Установка

Для проекта рекомендуется использовать виртуальное окружение Python.

### 1. Создание виртуального окружения

macOS / Linux: python3 -m venv .venv

Windows: python -m venv .venv

### 2. Активация виртуального окружения

macOS / Linux: source .venv/bin/activate

Windows: .venv\Scripts\activate


### 3. Установка зависимостей

pip install -r requirements.txt

# Часть 1. ML-модели

## Определение темы текста

Запуск: python text/text.py

Запуск собственного тестирования: python text/evaluate.py

## Распознавание голосовых команд

Запуск: python audio/audio.py

Тестовые аудиофайлы находятся в: audio/commands/

Запуск собственного тестирования: python audio/evaluate.py

## Классификация изображений

Используется `MobileNetV2` с предобученными весами ImageNet.

Запуск: python image/image.py

Тестовые изображения находятся в: image/test_images/

Запуск собственного тестирования: python image/evaluate.py

## Распознавание действий на видео

Запуск: python video/video.py

Тестовые видео находятся в: video/test_videos/

# Часть 2. FastAPI

## Запуск API

Из корневой директории проекта: python -m uvicorn api.main:app --reload

# Тестирование API

Для тестирования используются `pytest` и `FastAPI TestClient`.

Запуск всех тестов API из корневой директории: python -m pytest api/tests -v

В проекте реализовано 13 автоматических тестов, проверяющих успешные запросы, структуру ответа и обработку некорректных входных данных.

# CI

Для проекта настроен GitHub Actions.

При отправке изменений в репозиторий автоматически:

1. подготавливается Python-окружение
2. устанавливаются зависимости из `requirements.txt`
3. запускаются тесты `pytest`

Таким образом, работоспособность API автоматически проверяется после изменений в репозитории.

---

# Используемые технологии

- Python
- PyTorch
- TensorFlow / Keras
- Hugging Face Transformers
- torchvision
- OpenCV
- FastAPI
- Pydantic
- pytest
- GitHub Actions
