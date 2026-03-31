"""
attendance.py — Real-time face recognition and attendance logging.
Usage: python attendance.py [--threshold 80] [--output attendance.csv]
"""

import cv2
import os
import csv
import pickle
import argparse
from datetime import datetime

MODEL_DIR    = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH   = os.path.join(MODEL_DIR, "lbph_model.yml")
LABEL_PATH   = os.path.join(MODEL_DIR, "label_map.pkl")
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "attendance_logs")

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run train.py first."
        )
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    with open(LABEL_PATH, "rb") as f:
        label_map = pickle.load(f)
    return recognizer, label_map


def parse_label(raw_label: str):
    """Convert 'STU001_John_Doe' → ('STU001', 'John Doe')."""
    parts = raw_label.split("_", 1)
    student_id = parts[0] if len(parts) > 1 else "UNKNOWN"
    name = parts[1].replace("_", " ") if len(parts) > 1 else raw_label
    return student_id, name


def run_attendance(confidence_threshold: float = 80.0, output_file: str = None):
    recognizer, label_map = load_model()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if output_file is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_file = os.path.join(OUTPUT_DIR, f"attendance_{date_str}.csv")

    # Initialise CSV (write header only if file is new)
    file_is_new = not os.path.exists(output_file)
    csv_file = open(output_file, "a", newline="")
    writer = csv.writer(csv_file)
    if file_is_new:
        writer.writerow(["Student ID", "Name", "Date", "Time", "Confidence"])

    marked_today = set()  # avoid duplicate entries in one session

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("[INFO] Attendance system running. Press Q to quit.")
    print(f"[INFO] Logging to: {output_file}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)   # improve recognition under varying light
        faces = FACE_CASCADE.detectMultiScale(gray_eq, scaleFactor=1.2, minNeighbors=6)

        for (x, y, w, h) in faces:
            face_roi = cv2.resize(gray_eq[y:y + h, x:x + w], (200, 200))
            label_id, confidence = recognizer.predict(face_roi)

            # LBPH: lower confidence = better match
            is_recognized = confidence < confidence_threshold
            raw_label = label_map.get(label_id, "Unknown")
            student_id, name = parse_label(raw_label)

            if is_recognized:
                display_text = f"{name} ({student_id})"
                box_color = (0, 200, 0)

                if student_id not in marked_today:
                    now = datetime.now()
                    writer.writerow([
                        student_id,
                        name,
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S"),
                        f"{confidence:.1f}"
                    ])
                    csv_file.flush()
                    marked_today.add(student_id)
                    print(f"  [✓] Marked present: {name} ({student_id})  confidence={confidence:.1f}")
            else:
                display_text = "Unknown"
                box_color = (0, 0, 200)

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            label_bg_y = y - 30 if y > 30 else y + h + 10
            cv2.rectangle(frame, (x, label_bg_y), (x + w, label_bg_y + 25), box_color, -1)
            cv2.putText(frame, display_text, (x + 4, label_bg_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # HUD
        cv2.putText(frame, f"Present today: {len(marked_today)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Smart Attendance System — Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Session ended. {len(marked_today)} student(s) marked present.")
    print(f"[INFO] Log saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run real-time face attendance.")
    parser.add_argument("--threshold", type=float, default=80.0,
                        help="LBPH confidence threshold (lower = stricter). Default: 80")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output CSV file. Defaults to attendance_logs/attendance_YYYY-MM-DD.csv")
    args = parser.parse_args()

    run_attendance(args.threshold, args.output)
