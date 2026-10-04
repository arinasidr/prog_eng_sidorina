from pathlib import Path
import numpy as np
from keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from keras.utils import load_img, img_to_array

model = MobileNetV2(weights='imagenet')

image_path = input("Введите путь к изображению: ").strip()
path = Path(image_path)

if not path.exists():
    print("Изображение не найдено")
    raise SystemExit

image = load_img(
    path,
    target_size=(224, 224),
)

image_array = img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)
image_array = preprocess_input(image_array)

predictions = model.predict(image_array)

results = decode_predictions(
    predictions,
    top=5
)[0]

best_result = results[0]

print(f"\nОпределённый класс: {best_result[1]}")
print(f"Уверенность: {best_result[2]:.2%}")

print("\nТоп-5 вариантов:")
for _, label, score in results:
    print(f"{label}: {score:.2%}")
