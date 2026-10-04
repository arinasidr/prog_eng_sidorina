from transformers import pipeline

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

classifier = pipeline(
    "zero-shot-classification", 
    model=MODEL_NAME
)

text = input("Введите текст для определения темы: ")

default_categories = [
    "технологии",
    "спорт",
    "бизнес",
    "образование",
]

categories_input = input(
    "Введите категории через запятую или нажмите Enter для стандартных: "
)

if categories_input.strip():
    categories = [
        category.strip()
        for category in categories_input.split(",")
    ]
else:
    categories = default_categories

result = classifier(
    text,
    categories,
    multi_label=False, #считаем, что у текста одна основная тема, а не несколько независимых тем
)

print(f"\nТекст: {text}")
print(f"Определённая тема: {result['labels'][0]}")
print(f"Уверенность: {result['scores'][0]:.2%}")

print("\nВсе категории:")

for label, score in zip(result["labels"], result["scores"]):
    print(f"{label}: {score:.2%}")