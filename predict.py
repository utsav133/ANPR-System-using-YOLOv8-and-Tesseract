from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model(
    "car.jpeg",
    conf=0.4,
    save=True,
    show=True
)

print("Prediction completed")