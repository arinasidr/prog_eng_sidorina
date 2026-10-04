from pathlib import Path
import time

from transformers import pipeline


MODEL_NAME = "MIT/ast-finetuned-speech-commands-v2"

classifier = pipeline(
    "audio-classification",
    model=MODEL_NAME,
)

commands_dir = Path(__file__).resolve().parent / "commands"

audio_files = list(commands_dir.glob("*/*.wav"))

correct = 0
total_time = 0

for audio_file in audio_files:
    expected_label = audio_file.parent.name

    start_time = time.perf_counter()

    result = classifier(
        str(audio_file),
        top_k=1,
    )

    end_time = time.perf_counter()

    predicted_label = result[0]["label"]
    score = result[0]["score"]
    inference_time = end_time - start_time

    total_time += inference_time

    if predicted_label == expected_label:
        correct += 1
        status = "Yes"
    else:
        status = "No"

    print(f"\n{status} {audio_file.name}")
    print(f"Ожидалось: {expected_label}")
    print(f"Получено:   {predicted_label}")
    print(f"Уверенность: {score:.2%}")
    print(f"Время: {inference_time:.3f} сек.")

accuracy = correct / len(audio_files)
average_time = total_time / len(audio_files)

print("\n--------------------------")
print(f"Проверено файлов: {len(audio_files)}")
print(f"Правильно: {correct}/{len(audio_files)}")
print(f"Accuracy: {accuracy:.2%}")
print(f"Среднее время inference: {average_time:.3f} сек.")