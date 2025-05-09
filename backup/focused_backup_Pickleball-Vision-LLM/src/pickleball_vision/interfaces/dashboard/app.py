import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from typing import Dict, Any
from ..config.config import Config
from ..processors.video_processor import VideoProcessor
from ..utils.analyzer import VideoAnalyzer

def load_video_stats(stats_path: str) -> Dict[str, Any]:
    """Load video statistics from JSON file."""
    with open(stats_path, 'r') as f:
        return json.load(f)

def plot_detection_counts(stats: Dict[str, Any]):
    """Plot detection counts by class."""
    detections = stats['detections']['by_class']
    df = pd.DataFrame({
        'Class': list(detections.keys()),
        'Count': list(detections.values())
    })
    
    fig = px.bar(df, x='Class', y='Count', title='Detection Counts by Class')
    st.plotly_chart(fig)

def plot_motion_over_time(stats: Dict[str, Any]):
    """Plot motion over time."""
    motion_data = stats['motion']['motion_by_frame']
    df = pd.DataFrame(motion_data)
    
    fig = px.line(df, x='frame', y='motion', title='Motion Over Time')
    st.plotly_chart(fig)

def plot_detection_heatmap(stats: Dict[str, Any]):
    """Plot detection heatmap."""
    detections = stats['detections']['by_frame']
    
    # Create frame-by-class matrix
    classes = set()
    for frame in detections:
        for det in frame['detections']:
            classes.add(det['class'])
    
    classes = sorted(list(classes))
    frames = range(len(detections))
    
    # Initialize heatmap data
    heatmap_data = np.zeros((len(classes), len(frames)))
    
    # Fill heatmap data
    for i, frame in enumerate(detections):
        for det in frame['detections']:
            class_idx = classes.index(det['class'])
            heatmap_data[class_idx, i] = det['confidence']
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=frames,
        y=classes,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title='Detection Confidence Heatmap',
        xaxis_title='Frame',
        yaxis_title='Class'
    )
    
    st.plotly_chart(fig)

def main():
    st.title('Pickleball Vision Analysis Dashboard')
    
    # Sidebar
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Video Analysis', 'Real-time Processing'])
    
    if page == 'Video Analysis':
        # File uploader
        uploaded_file = st.file_uploader('Upload video file', type=['mp4', 'avi'])
        
        if uploaded_file is not None:
            # Save uploaded file
            video_path = Path('data/uploaded_videos') / uploaded_file.name
            video_path.parent.mkdir(exist_ok=True)
            
            with open(video_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Process video
            if st.button('Process Video'):
                with st.spinner('Processing video...'):
                    # Load config
                    cfg = Config()
                    
                    # Process video
                    processor = VideoProcessor(cfg)
                    processor.process_video(str(video_path))
                    
                    # Analyze video
                    analyzer = VideoAnalyzer(cfg)
                    stats = analyzer.analyze_video(str(video_path))
                    
                    # Save stats
                    stats_path = Path('data/analysis') / f'{video_path.stem}_stats.json'
                    stats_path.parent.mkdir(exist_ok=True)
                    
                    with open(stats_path, 'w') as f:
                        json.dump(stats, f, indent=2)
                    
                    st.success('Video processing complete!')
            
            # Load and display stats if available
            stats_path = Path('data/analysis') / f'{video_path.stem}_stats.json'
            if stats_path.exists():
                stats = load_video_stats(str(stats_path))
                
                # Display video info
                st.header('Video Information')
                info = stats['video_info']
                col1, col2, col3 = st.columns(3)
                col1.metric('Resolution', f"{info['width']}x{info['height']}")
                col2.metric('FPS', f"{info['fps']:.1f}")
                col3.metric('Duration', f"{info['duration']:.1f}s")
                
                # Display plots
                st.header('Analysis Results')
                plot_detection_counts(stats)
                plot_motion_over_time(stats)
                plot_detection_heatmap(stats)
                
                # Display sample frames
                st.header('Sample Frames')
                frames_dir = Path('data/frames')
                if frames_dir.exists():
                    frames = sorted(frames_dir.glob('*.jpg'))[:5]
                    cols = st.columns(len(frames))
                    for col, frame_path in zip(cols, frames):
                        col.image(str(frame_path), caption=frame_path.name)
    
    else:  # Real-time Processing
        st.header('Real-time Video Processing')
        
        # Camera input
        camera_input = st.camera_input('Take a picture')
        
        if camera_input is not None:
            # Convert to numpy array
            bytes_data = camera_input.getvalue()
            nparr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process frame
            if st.button('Process Frame'):
                with st.spinner('Processing frame...'):
                    # Load config
                    cfg = Config()
                    
                    # Process frame
                    processor = VideoProcessor(cfg)
                    detections = processor.detector.detect(frame)
                    
                    # Visualize detections
                    output_frame = processor.visualizer.draw_detections(frame, detections)
                    
                    # Display results
                    st.image(output_frame, channels='BGR', caption='Processed Frame')
                    
                    # Display detection info
                    st.subheader('Detections')
                    for det in detections:
                        st.write(f"- {det['class']} (confidence: {det['confidence']:.2f})")

if __name__ == '__main__':
    main() 