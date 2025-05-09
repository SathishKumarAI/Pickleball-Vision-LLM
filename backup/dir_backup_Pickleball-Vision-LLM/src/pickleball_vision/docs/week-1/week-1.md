
# 📅 Week 1 Plan: Data, Vision, and Dev Setup for Project Pickleball V Couch (Pickleball LLM)

---

## 🎯 Objectives

- Establish the full development environment (Git, GitHub, Conda, Python, VSCode)  
- Create a scalable system architecture  
- Begin collecting and preprocessing data (videos and frames)  
- Run a test detection with YOLOv8  
- Set up data versioning using DVC  

---

## 🧰 1. Developer Environment Setup

### ✅ Git + GitHub Setup

- Create a GitHub repository: `pickleball-llm`  
- Clone locally and initialize Git:

```bash
git init
git remote add origin <your-repo-url>
```

- Add a `.gitignore`

---

### ✅ VSCode Setup

Use the following VSCode extensions:

- Python  
- Pylance  
- GitLens  
- Jupyter  
- Docker *(optional)*  

Create a `launch.json` for debugging and `settings.json` to configure linting.

---

## 🐍 2. Python + Conda Environment

### ✅ Create and Activate Environment

```bash
conda create -n pickleball-llm python=3.11
conda activate pickleball-llm
```

### ✅ Install Core Libraries

```bash
pip install opencv-python-headless ultralytics ffmpeg-python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install dvc mlflow pandas tqdm
pip install jupyterlab
```

### ✅ Save Environment

```bash
conda env export > environment.yml
```

---

## 🧱 3. Project Structure (System Design – MVP Scope)

```plaintext
pickleball-llm/
├── data/
│   ├── raw_videos/
│   └── frames/
├── models/
├── notebooks/
├── src/
│   ├── data_collection/
│   ├── preprocessing/
│   ├── vision/
│   └── utils/
├── scripts/
├── .dvc/
├── .gitignore
├── README.md
├── environment.yml
└── requirements.txt
```

---

### ✅ System Design Diagram (High-Level MVP Pipeline)

```plaintext
YouTube Videos 
   ⬇
[Video Scraper]  --> [Frame Extractor] --> [Vision Models]
                                      ⬇
                           [YOLOv8, SAM, MediaPipe]
                                      ⬇
                          [Structured JSON Game State]
                                      ⬇
                                 [LLM Prompting]
                                      ⬇
                          [Strategy Feedback / Coaching]
```

---

## 📽️ 4. Data Collection

### ✅ Scrape Videos

```bash
yt-dlp -f "bestvideo+bestaudio" -o "data/raw_videos/%(title)s.%(ext)s" <YouTube_URL>
```

### ✅ Extract Frames

```python
import cv2
import os

def extract_frames(video_path, output_folder, interval=5):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / interval)

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_interval == 0:
            frame_name = os.path.join(output_folder, f"frame_{i}.jpg")
            cv2.imwrite(frame_name, frame)
        i += 1
    cap.release()
```

---

## 🎯 5. YOLOv8 Test Run

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("data/frames/frame_001.jpg")
results.show()
```

---

## 📦 6. DVC Setup

```bash
dvc init
dvc add data/raw_videos
dvc add data/frames
git add .dvc .gitignore
git commit -m "Initialize data pipeline and tracking"
```

---

## ✅ Deliverables by End of Week 1

| Deliverable       | Description                                     |
|-------------------|-------------------------------------------------|
| 🗂️ Project Structure | Proper folders, code modularity                 |
| 🐍 Conda Environment | Reproducible with `environment.yml`            |
| 🎞️ Video & Frames   | 10–15 videos, 1000+ frames extracted           |
| 🎯 YOLO Detection   | Working sample output with bounding boxes      |
| 📊 DVC Tracking     | Raw + processed data under version control     |
| 🌐 GitHub Repo      | Initial commit pushed and documented           |
```