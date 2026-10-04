from pathlib import Path
import time

import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.utils import load_img, img_to_array


model = MobileNetV2(weights="imagenet")

images_dir = Path(__file__).resolve().parent / "test_images"

image_files = []

for extension in ("*.jpg", "*.jpeg", "*.png"):
    image_files.extend(images_dir.glob(f"*/{extension}"))

if not image_files:
    print("Тестовые изображения не найдены.")
    raise SystemExit

correct = 0
total_time = 0

for image_file in image_files:
    expected_label = image_file.parent.name

    image = load_img(
        image_file,
        target_size=(224, 224),
    )

    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    start_time = time.perf_counter()

    predictions = model.predict(
        image_array,
        verbose=0,
    )

    end_time = time.perf_counter()

    result = decode_predictions(
        predictions,
        top=1,
    )[0][0]

    predicted_label = result[1]
    score = result[2]

    inference_time = end_time - start_time
    total_time += inference_time

    if predicted_label == expected_label:
        correct += 1
        status = "Yes"
    else:
        status = "No"

    print(f"\n{status} {image_file.name}")
    print(f"Ожидалось: {expected_label}")
    print(f"Получено:   {predicted_label}")
    print(f"Уверенность: {score:.2%}")
    print(f"Время inference: {inference_time:.3f} сек.")

accuracy = correct / len(image_files)
average_time = total_time / len(image_files)

print(f"Проверено изображений: {len(image_files)}")
print(f"Правильно: {correct}/{len(image_files)}")
print(f"Accuracy: {accuracy:.2%}")
print(f"Среднее время inference: {average_time:.3f} сек.")