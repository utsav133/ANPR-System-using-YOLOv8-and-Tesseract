import cv2
import pytesseract
import numpy as np
import re
import csv
from datetime import datetime
from ultralytics import YOLO

# ==========================
# TESSERACT PATH
# ==========================
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ==========================
# LOAD MODEL
# ==========================
model = YOLO("runs/detect/train/weights/best.pt")

# ==========================
# LOAD IMAGE
# ==========================
image = cv2.imread("car.jpeg")

if image is None:
    print("Error: car.jpeg not found")
    exit()

# ==========================
# DETECTION
# ==========================
results = model(image, conf=0.4)

if len(results[0].boxes) == 0:
    print("No license plate detected")
    exit()

# ==========================
# PROCESS FIRST DETECTED PLATE
# ==========================
for box in results[0].boxes:

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    confidence = float(box.conf[0])

    # Add padding around plate
    pad = 10

    h, w = image.shape[:2]

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    # Crop plate
    plate = image[y1:y2, x1:x2]

    if plate.size == 0:
        print("Invalid crop")
        exit()

    cv2.imwrite("cropped_plate.jpg", plate)

    # ==========================
    # OCR PREPROCESSING
    # ==========================
    gray = cv2.cvtColor(
        plate,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=8,
        fy=8,
        interpolation=cv2.INTER_CUBIC
    )

    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])

    gray = cv2.filter2D(
        gray,
        -1,
        kernel
    )

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    cv2.imwrite(
        "processed_plate.jpg",
        thresh
    )

    # ==========================
    # OCR
    # ==========================
    text = pytesseract.image_to_string(
        thresh,
        config="--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    plate_number = re.sub(
        r"[^A-Z0-9]",
        "",
        text.upper()
    )

    # ==========================
    # PRINT RESULT
    # ==========================
    print(f"Detected Plate: {plate_number}")
    print(f"Confidence: {confidence:.3f}")

    # ==========================
    # CSV LOGGING
    # ==========================
    with open(
        "results.csv",
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            plate_number,
            round(confidence, 3)
        ])

    # Save final image
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        plate_number,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imwrite(
        "final_result.jpg",
        image
    )

    # Stop after first detection
    break

print("Result saved:")
print(" - final_result.jpg")
print(" - cropped_plate.jpg")
print(" - processed_plate.jpg")
print(" - results.csv")