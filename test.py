from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model(
    source="test/images",
    conf=0.4,
    save=True
)

print("Done")