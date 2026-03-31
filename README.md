# Smart Attendance System using Face Recognition

A real-time, webcam-based attendance tracking system built with **Python** and **OpenCV**. It uses **LBPH (Local Binary Pattern Histograms)** face recognition — a classical computer vision technique — to identify registered individuals and automatically log their attendance to a timestamped CSV file.

---

## Problem Statement

Manual attendance in classrooms or offices is repetitive, time-consuming, and prone to errors such as proxy attendance. This project automates the process using a webcam — no special hardware required.

---

## Features

- Register any number of people via webcam
- Train a local LBPH face recognition model on registered faces
- Run real-time recognition from a live webcam feed
- Automatically log attendance (name, ID, date, time) to a CSV file
- Each person is marked **once per session** (no duplicates)
- View daily or all-time attendance reports from the terminal
- Adjustable confidence threshold for stricter/looser recognition

---

## Project Structure

```
smart-attendance/
├── src/
│   ├── register.py       # Step 1 — Register a new face
│   ├── train.py          # Step 2 — Train the recognizer
│   ├── attendance.py     # Step 3 — Run live attendance
│   └── view_report.py    # Step 4 — View attendance logs
├── dataset/              # Auto-created face sample images
├── model/                # Auto-created trained model files
├── attendance_logs/      # Auto-created daily CSV logs
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.8 or higher
- A working webcam

### Installation

```bash
# 1. Clone the repository
https://github.com/khushiepatell/Smart-Attendance-System-using-Face-Recognition.git

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** `opencv-contrib-python` is required for LBPH face recognition. Do **not** install both `opencv-python` and `opencv-contrib-python` simultaneously — uninstall one first if there's a conflict.

---

## Usage

### Step 1 — Register a Face

```bash
python src/register.py --name "Jane Smith" --id STU001
```

- The webcam will open and capture **30 face samples** automatically.
- Keep your face clearly visible and slightly move around for variety.
- Samples are saved to `dataset/STU001_Jane_Smith/`.

Optional: change number of samples:
```bash
python src/register.py --name "Jane Smith" --id STU001 --samples 50
```

### Step 2 — Train the Model

Run this after registering all people (or whenever you add someone new):

```bash
python src/train.py
```

The trained model is saved to `model/lbph_model.yml`.

### Step 3 — Take Attendance

```bash
python src/attendance.py
```

- The webcam opens and recognizes faces in real time.
- A green box = recognized, red box = unknown.
- Each recognized person is logged **once** per session to `attendance_logs/attendance_YYYY-MM-DD.csv`.
- Press **Q** to quit.

Optional: adjust recognition sensitivity:
```bash
python src/attendance.py --threshold 70   # stricter (smaller = stricter)
```

### Step 4 — View Attendance Logs

```bash
# Today's log
python src/view_report.py

# Specific date
python src/view_report.py --date 2026-03-31

# All logs
python src/view_report.py --all
```

---

## How It Works

1. **Face Detection** — OpenCV's Haar Cascade classifier detects face regions in each video frame.
2. **Image Segmentation** — The detected face region (ROI) is cropped and isolated from the background.
3. **Feature Extraction** — LBPH encodes each face as a histogram of Local Binary Pattern textures — a compact, illumination-robust descriptor.
4. **Recognition** — The descriptor is compared against all trained models. The closest match (lowest LBPH confidence score) below the threshold is accepted.
5. **Logging** — A match triggers a timestamped CSV entry, recorded once per session per person.

---

## Sample Output

**Terminal:**
```
[✓] Marked present: Jane Smith (STU001)  confidence=42.3
[✓] Marked present: Rahul Verma (STU002)  confidence=55.1
```

**attendance_logs/attendance_2026-03-31.csv:**
```
Student ID,Name,Date,Time,Confidence
STU001,Jane Smith,2026-03-31,09:02:14,42.3
STU002,Rahul Verma,2026-03-31,09:03:07,55.1
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'cv2.face'` | Run `pip install opencv-contrib-python` |
| Webcam not detected | Check camera index; try `cv2.VideoCapture(1)` in attendance.py |
| Too many "Unknown" detections | Increase `--threshold` (e.g., 90) or register more samples |
| Model not found error | Make sure you ran `train.py` after `register.py` |

---

## Course Context

This project was built as part of a **Computer Vision** course capstone (BYOP). It applies:
- **Image segmentation** — isolating the face ROI from full frames
- **Feature extraction** — LBPH texture descriptors
- **Feature matching** — nearest-neighbour matching against trained descriptors
- **OpenCV** — for detection, recognition, and video I/O

---

## License

MIT License — free to use and modify.
