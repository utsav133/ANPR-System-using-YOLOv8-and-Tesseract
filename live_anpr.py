import cv2
import pytesseract
import numpy as np
import re
import csv
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# TESSERACT PATH
# ==========================================
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ==========================================
# LOAD YOLO MODEL
# ==========================================
model = YOLO("runs/detect/train/weights/best.pt")

# ==========================================
# LOAD VIDEO
# ==========================================
cap = cv2.VideoCapture("demo.mp4")

# Store unique detected plates
seen_plates = set()

# ==========================================
# PROCESS VIDEO
# ==========================================
while True:

    ret, frame = cap.read()

    # Stop automatically when video ends
    if not ret:
        print("\nVideo completed.")
        break

    results = model(frame, conf=0.4)

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])

        # Add padding around plate
        pad = 10

        h, w = frame.shape[:2]

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        plate = frame[y1:y2, x1:x2]

        if plate.size == 0:
            continue

        # ==========================================
        # OCR PREPROCESSING
        # ==========================================
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

        # ==========================================
        # OCR
        # ==========================================
        text = pytesseract.image_to_string(
            thresh,
            config="--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        plate_number = re.sub(
            r"[^A-Z0-9]",
            "",
            text.upper()
        )

        if len(plate_number) < 4:
            continue

        # ==========================================
        # SAVE UNIQUE PLATES ONLY
        # ==========================================
        if plate_number not in seen_plates:

            seen_plates.add(plate_number)

            print(
                f"Detected Plate: {plate_number} | Confidence: {confidence:.3f}"
            )

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

        # ==========================================
        # DRAW RESULTS
        # ==========================================
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            plate_number,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # ==========================================
    # DISPLAY VIDEO
    # ==========================================
    cv2.imshow(
        "ANPR Video",
        frame
    )

    # Optional: ESC to stop early
    if cv2.waitKey(1) & 0xFF == 27:
        print("\nStopped by user.")
        break

# ==========================================
# CLEANUP
# ==========================================
cap.release()
cv2.destroyAllWindows()

print("\nProgram stopped successfully.")