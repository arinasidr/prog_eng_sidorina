from pathlib import Path

import cv2
import torch
from torchvision.models.video import r3d_18, R3D_18_Weights

NUM_FRAMES = 16
weights = R3D_18_Weights.KINETICS400_V1

model = r3d_18(weights=weights)
model.eval()

preprocess = weights.transforms()
categories = weights.meta['categories']

video_path = input("Введите путь к видео: ").strip()
path = Path(video_path)

if not path.exists():
    print("Видеофайл не найден")
    raise SystemExit

cap = cv2.VideoCapture(str(path))
if not cap.isOpened():
    print("Не удалось открыть видео")
    raise SystemExit

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

start_frame = max(
    0,
    frame_count // 2 - NUM_FRAMES // 2,
)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frames = []

for _ in range(NUM_FRAMES):
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_tensor = torch.from_numpy(frame)
    frame_tensor = frame_tensor.permute(2, 0, 1)

    frames.append(frame_tensor)

cap.release()

if not frames:
    print("не удалось прочитать кадры видео")
    raise SystemExit

while len(frames) < NUM_FRAMES:
    frames.append(frames[-1].clone())

video = torch.stack(frames)
batch = preprocess(video).unsqueeze(0)

with torch.inference_mode():
    prediction = model(batch).squeeze(0)
    probabilities = prediction.softmax(dim=0)

top_probabilities, top_indices = probabilities.topk(5)
print("\nТоп-5 действий:")

for probability, index in zip(top_probabilities, top_indices):
    label = categories[index.item()]
    score = probability.item()

    print(f"{label}: {score:.2%}")
