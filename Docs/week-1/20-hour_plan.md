**Week 1 Plan** (20-hour, weekend-weighted schedule) 

```markdown
# 🗓️ 20-Hour Week 1 Plan – Project Pickleball V Couch (Pickleball LLM)

## 🎯 Goal
Set up your environment, collect YouTube match data, extract frames, run YOLOv8 detection, and track data with DVC — all within 20 hours total for the week.

---

### 📍 Start Date
**Monday**

### ⏳ Total Hours
**20 hrs**

### ⚖️ Distribution
Light weekdays, deeper weekend focus

---

## ✅ Monday (2 hrs) – Kickoff & Environment Setup

- Create GitHub repo & initialize `.git` (20 min)  
- Set up Conda environment + install key libraries (60 min)  
- Create folder structure (`/data`, `/src`, `/models`, etc.) (20 min)  
- Open in VSCode, configure Python interpreter + Git integration (20 min)  

**Total: 2 hrs**

---

## ✅ Tuesday (2 hrs) – Download Match Videos

- Find 5–6 good YouTube pickleball matches (30 min)  
- Use `yt-dlp` to download them (automate in background) (30 min)  
- Organize files into `/data/raw_videos/` (30 min)  
- Create a CSV to log match metadata (20 min)  

**Total: 2 hrs**

---

## ✅ Wednesday (2 hrs) – Extract Video Frames

- Write basic OpenCV script to extract 5 FPS frames (30 min)  
- Run it on 2–3 downloaded videos (1 hr)  
- Save output in organized `/data/frames/<video_id>/` folders (30 min)  

**Total: 2 hrs**

---

## ✅ Thursday (2 hrs) – YOLOv8 Detection Test

- Install `ultralytics` package and load YOLOv8 (30 min)  
- Run inference on 10–15 sample frames (30 min)  
- Save visual results + bounding box JSON output (1 hr)  

**Total: 2 hrs**

---

## ✅ Friday (2 hrs) – DVC + Code Cleanup

- Install & initialize DVC (15 min)  
- Track `data/raw_videos` + `data/frames` (30 min)  
- Commit `.dvc` files to GitHub (30 min)  
- Refactor code: move scripts into `/src` and clean up (45 min)  

**Total: 2 hrs**

---

## ✅ Saturday (5 hrs) – Integration & Full Pipeline

- Combine video download, frame extraction, and detection into `pipeline.py` (2 hrs)  
- Add logging + progress indicators with `tqdm` (30 min)  
- Run full pipeline on 2–3 matches and inspect outputs (1.5 hrs)  
- Push final code + DVC state to GitHub (1 hr)  

**Total: 5 hrs**

---

## ✅ Sunday (5 hrs) – Annotation + Documentation

- Set up Label Studio or CVAT locally (1 hr)  
- Manually annotate 50–100 frames with player and ball boxes (2 hrs)  
- Define JSON schema for gameplay state (30 min)  
- Update `README.md`: setup guide, usage steps, architecture diagram (1 hr)  
- Write Week 2 preview plan (30 min)  

**Total: 5 hrs**

---

## 🧭 Summary of Time Allocation

| Day      | Focus Area                    | Hours |
|----------|-------------------------------|-------|
| Monday   | Dev environment & Git setup   | 2 hrs |
| Tuesday  | Video scraping                | 2 hrs |
| Wednesday| Frame extraction              | 2 hrs |
| Thursday | YOLOv8 object detection       | 2 hrs |
| Friday   | DVC tracking + refactor       | 2 hrs |
| Saturday | Full pipeline integration     | 5 hrs |
| Sunday   | Annotation + docs + cleanup   | 5 hrs |
| **TOTAL**|                               | **20 hrs ✅** |
```
