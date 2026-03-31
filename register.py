"""
register.py — Register new faces into the attendance system.
Usage: python register.py --name "John Doe" --id STU001
"""

import cv2
import os
import argparse
import time

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def register_face(name: str, student_id: str, n_samples: int = 30):
    """Capture face samples and save them for training."""
    person_dir = os.path.join(DATASET_DIR, f"{student_id}_{name.replace(' ', '_')}")
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check your camera connection.")

    print(f"[INFO] Registering: {name} (ID: {student_id})")
    print(f"[INFO] Look at the camera. Capturing {n_samples} samples...")

    count = 0
    while count < n_samples:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face_roi, (200, 200))
            filepath = os.path.join(person_dir, f"sample_{count:03d}.jpg")
            cv2.imwrite(filepath, face_resized)
            count += 1

            # Draw feedback on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}/{n_samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Register Face — Press Q to quit early", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.1)  # slight delay between captures

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Registration complete. {count} samples saved to: {person_dir}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a new face for attendance.")
    parser.add_argument("--name", required=True, help="Full name of the person")
    parser.add_argument("--id", required=True, dest="student_id", help="Student/Employee ID")
    parser.add_argument("--samples", type=int, default=30, help="Number of samples to capture (default: 30)")
    args = parser.parse_args()

    register_face(args.name, args.student_id, args.samples)
