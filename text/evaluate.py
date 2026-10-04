from transformers import pipeline
import time

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

classifier = pipeline(
    "zero-shot-classification",
    model=MODEL_NAME,
)

categories = [
    "технологии",
    "спорт",
    "бизнес",
    "образование",
]

test_data = [
    ("Футбольная команда выиграла финальный матч чемпионата.", "спорт"),
    ("Теннисист одержал победу на международном турнире.", "спорт"),
    ("Спортсмен установил новый мировой рекорд по плаванию.", "спорт"),

    ("Компания представила новый процессор для ноутбуков.", "технологии"),
    ("Разработчики выпустили новую версию операционной системы.", "технологии"),
    ("Учёные создали новый алгоритм искусственного интеллекта.", "технологии"),

    ("Компания сообщила о росте прибыли за последний квартал.", "бизнес"),
    ("Банк объявил о запуске нового финансового продукта.", "бизнес"),
    ("Продажи компании выросли на двадцать процентов.", "бизнес"),

    ("Студенты начали новый учебный семестр в университете.", "образование"),
    ("Школа запустила новую программу изучения математики.", "образование"),
    ("Университет открыл новый курс по программированию.", "образование"),
]

correct = 0
total_time = 0

for text, expected_label in test_data:
    start_time = time.perf_counter()

    result = classifier(
        text,
        categories,
        multi_label=False,
    )

    end_time = time.perf_counter()

    predicted_label = result["labels"][0]
    inference_time = end_time - start_time

    total_time += inference_time

    if predicted_label == expected_label:
        correct += 1
        status = "Yes"
    else:
        status = "No"

    print(f"\n{status} {text}")
    print(f"Ожидалось: {expected_label}")
    print(f"Получено:   {predicted_label}")
    print(f"Уверенность: {result['scores'][0]:.2%}")
    print(f"Время: {inference_time:.3f} сек.")

accuracy = correct / len(test_data)
average_time = total_time / len(test_data)

print("\n--------------------------")
print(f"Правильно: {correct}/{len(test_data)}")
print(f"Accuracy: {accuracy:.2%}")
print(f"Среднее время inference: {average_time:.3f} сек.")