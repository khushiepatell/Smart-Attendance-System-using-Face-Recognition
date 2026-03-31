"""
train.py — Train the LBPH face recognizer on registered face samples.
Usage: python train.py
"""

import cv2
import os
import pickle
import numpy as np

DATASET_DIR   = os.path.join(os.path.dirname(__file__), "..", "dataset")
MODEL_DIR     = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH    = os.path.join(MODEL_DIR, "lbph_model.yml")
LABEL_PATH    = os.path.join(MODEL_DIR, "label_map.pkl")


def load_dataset():
    """Load all face images and build integer label map."""
    faces, labels = [], []
    label_map = {}   # int  -> "ID_Name"
    current_label = 0

    if not os.path.isdir(DATASET_DIR) or not os.listdir(DATASET_DIR):
        raise FileNotFoundError(
            f"No dataset found at '{DATASET_DIR}'. "
            "Run register.py first to add faces."
        )

    for person_folder in sorted(os.listdir(DATASET_DIR)):
        folder_path = os.path.join(DATASET_DIR, person_folder)
        if not os.path.isdir(folder_path):
            continue

        label_map[current_label] = person_folder  # e.g., "STU001_John_Doe"
        sample_count = 0

        for img_file in os.listdir(folder_path):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200, 200))
            faces.append(img)
            labels.append(current_label)
            sample_count += 1

        print(f"  Loaded {sample_count:>3d} samples — {person_folder}")
        current_label += 1

    return faces, np.array(labels), label_map


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("[INFO] Loading dataset...")
    faces, labels, label_map = load_dataset()
    print(f"[INFO] Total: {len(faces)} images across {len(label_map)} person(s).\n")

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )

    print("[INFO] Training LBPH model — this may take a moment...")
    recognizer.train(faces, labels)

    recognizer.save(MODEL_PATH)
    with open(LABEL_PATH, "wb") as f:
        pickle.dump(label_map, f)

    print(f"[INFO] Model saved  : {MODEL_PATH}")
    print(f"[INFO] Labels saved : {LABEL_PATH}")
    print("[INFO] Training complete!")


if __name__ == "__main__":
    train()
