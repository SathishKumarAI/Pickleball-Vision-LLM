import streamlit as st
import tempfile
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional
import json
import time
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from src.data_collection.frame_sampler import sample_frames_from_video
from src.models.pose_extractor import extract_pose_from_frame
from src.models.llm_clip_integration import analyze_gameplay_with_llm
from src.utils.visualizer import draw_pose_on_frame
from src.utils.video_processor import VideoProcessor
from src.models.shot_classifier import classify_shots
from src.utils.metrics_analyzer import calculate_performance_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom imports (adjust as needed)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

class PickleballAnalyzer:
    def __init__(self):
        self.setup_ui()
        self.video_processor = VideoProcessor()
        
    def setup_ui(self):
        """Setup the Streamlit UI components"""
        st.set_page_config(page_title="🏓 Pickleball Vision LLM", layout="wide")
        st.title("🏓 Advanced Pickleball Vision Analysis")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Analysis Settings")
            self.sampling_rate = st.slider("Frame Sampling Rate", 1, 60, 30)
            self.confidence_threshold = st.slider("Pose Confidence Threshold", 0.0, 1.0, 0.7)
            self.enable_shot_detection = st.checkbox("Enable Shot Detection", True)
            self.enable_performance_metrics = st.checkbox("Enable Performance Metrics", True)
            
            st.header("📊 Analysis History")
            if st.session_state.analysis_history:
                for idx, analysis in enumerate(st.session_state.analysis_history):
                    if st.button(f"View Analysis {idx + 1} - {analysis['timestamp']}"):
                        st.session_state.current_analysis = analysis

    @st.cache_data
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Process video with caching for better performance"""
        try:
            results = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "frames": [],
                "poses": [],
                "shots": [],
                "metrics": {},
                "analysis": None
            }

            # Sample frames
            with st.spinner("📸 Sampling frames..."):
                sampled_frames = sample_frames_from_video(video_path, every_nth=self.sampling_rate)
                results["frames"] = sampled_frames

            # Process poses in parallel
            with st.spinner("🧍 Extracting poses..."):
                with ThreadPoolExecutor() as executor:
                    results["poses"] = list(executor.map(
                        lambda frame: extract_pose_from_frame(frame, self.confidence_threshold),
                        sampled_frames
                    ))

            # Shot detection
            if self.enable_shot_detection:
                with st.spinner("🎾 Detecting shots..."):
                    results["shots"] = classify_shots(results["poses"])

            # Performance metrics
            if self.enable_performance_metrics:
                with st.spinner("📊 Calculating metrics..."):
                    results["metrics"] = calculate_performance_metrics(
                        results["poses"],
                        results["shots"]
                    )

            # LLM Analysis
            with st.spinner("🧠 Generating AI insights..."):
                results["analysis"] = analyze_gameplay_with_llm(
                    video_path,
                    poses=results["poses"],
                    shots=results["shots"],
                    metrics=results["metrics"]
                )

            return results

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            st.error(f"Error processing video: {str(e)}")
            return None

    def display_results(self, results: Dict[str, Any]):
        """Display analysis results in an organized manner"""
        if not results:
            return

        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎾 Shot Analysis", "📈 Metrics", "🎥 Frame Analysis"])

        with tab1:
            st.subheader("📝 Analysis Summary")
            st.markdown(results["analysis"]["summary"])
            
            st.subheader("🎯 Key Observations")
            for obs in results["analysis"].get("observations", []):
                st.markdown(f"- {obs}")

        with tab2:
            if results["shots"]:
                shot_df = pd.DataFrame(results["shots"])
                st.subheader("Shot Distribution")
                st.bar_chart(shot_df["type"].value_counts())
                
                st.subheader("Shot Timeline")
                st.line_chart(shot_df["confidence"])

        with tab3:
            if results["metrics"]:
                st.subheader("Performance Metrics")
                metrics_df = pd.DataFrame([results["metrics"]])
                st.dataframe(metrics_df)

        with tab4:
            st.subheader("Frame Analysis")
            selected_frame = st.slider("Select Frame", 0, len(results["frames"]) - 1)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(
                    Image.fromarray(cv2.cvtColor(results["frames"][selected_frame], cv2.COLOR_BGR2RGB)),
                    caption="Original Frame"
                )
            with col2:
                annotated_frame = draw_pose_on_frame(
                    results["frames"][selected_frame].copy(),
                    results["poses"][selected_frame]
                )
                st.image(
                    Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)),
                    caption="Annotated Frame"
                )

    def run(self):
        """Main application loop"""
        uploaded_file = st.file_uploader("Upload Pickleball video (.mp4)", type=["mp4"])

        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name

            st.video(video_path)
            
            if st.button("🎬 Start Analysis"):
                try:
                    results = self.process_video(video_path)
                    if results:
                        st.session_state.analysis_history.append(results)
                        st.session_state.current_analysis = results
                        self.display_results(results)
                        
                        # Export results
                        if st.download_button(
                            "📥 Download Analysis Report",
                            json.dumps(results, default=str),
                            "analysis_report.json"
                        ):
                            st.success("Report downloaded successfully!")
                            
                finally:
                    # Cleanup
                    os.unlink(video_path)

if __name__ == "__main__":
    analyzer = PickleballAnalyzer()
    analyzer.run()