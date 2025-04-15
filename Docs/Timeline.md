# 🏓 5-Week MVP Plan for Project Pickleball V Couch (Pickleball LLM)

## 📅 Goal
Build a vision + LLM-powered coaching assistant using AI and MLOps foundations.

---

## ✅ Week 1: Data Collection & Vision Pipeline Setup

- Set up project directory and initialize Git repository  
- Use `yt-dlp` to scrape 10–15 YouTube pickleball match videos  
- Extract frames at 5 FPS using OpenCV or FFmpeg  
- Integrate YOLOv8 for player, ball, and net detection  
- Set up DVC for tracking data versions  

---

## ✅ Week 2: Vision Enhancements + Pose Estimation

- Integrate MediaPipe Pose for player skeletal key points  
- Use SAM (Segment Anything) to segment court area  
- Design structured JSON schema to store game states  
- Annotate ~200 frames using Label Studio or CVAT  
- Add game context labels (serve, rally, smash) manually or semi-automatically  
- Track detection performance in MLflow (accuracy, speed)  

---

## ✅ Week 3: LLM Integration & Prompt Design

- Integrate BLIP-2 or CLIP to caption gameplay frames  
- Create structured prompts using JSON game states  
- Set up and test Mistral or Llama-2 LLM for feedback generation  
- Fine-tune or instruct-tune LLM on 20–30 strategy examples  
- Log prompts, outputs, and evaluation scores in MLflow  
- Use DVC to version LLM input/output datasets  

---

## ✅ Week 4: MVP Assembly & Streamlit Interface

- Build Streamlit or CLI tool for full pipeline execution  
- Integrate frame visualization and JSON game state display  
- Connect LLM output to the front-end for coaching feedback  
- Dockerize full application including vision + LLM models  
- Run full tests on 3 full-length matches to validate pipeline  

---

## ✅ Week 5: MLOps, Testing, and Final Polish

- Set up CI/CD (GitHub Actions) to test pipeline modules  
- Track LLM and vision experiments in MLflow with tags and notes  
- Write final `pipeline.py` orchestrator script for local runs  
- Prepare 500–1000 frame annotated dataset  
- Record and finalize MVP walkthrough demo video  
