from pathlib import Path

from transformers import pipeline


MODEL_NAME = "MIT/ast-finetuned-speech-commands-v2"

classifier = pipeline(
    "audio-classification",
    model=MODEL_NAME,
)

audio_path = input("Введите путь к аудиофайлу: ").strip()

path = Path(audio_path)

if not path.exists():
    print("Аудиофайл не найден.")
    raise SystemExit

result = classifier(
    str(path),
    top_k=5,
)

best_result = result[0]

print(f"\nОпределённая команда: {best_result['label']}")
print(f"Уверенность: {best_result['score']:.2%}")

print("\nТоп-5 вариантов:")

for item in result:
    print(f"{item['label']}: {item['score']:.2%}")