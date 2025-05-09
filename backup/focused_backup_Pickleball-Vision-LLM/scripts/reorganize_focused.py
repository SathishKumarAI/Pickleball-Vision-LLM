import os
import shutil
from pathlib import Path

def ensure_directory_exists(path):
    """Ensure directory exists before writing file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def create_core_structure():
    """Create the core project structure."""
    directories = [
        # Core ML components
        "src/vision",          # Computer vision components
        "src/llm",            # Language model components
        "src/fusion",         # Vision-LLM fusion logic
        
        # Data handling
        "data/videos",        # Raw video data
        "data/annotations",   # Game annotations
        "data/models",        # Trained models
        
        # API and deployment
        "src/api",            # API endpoints
        "src/web",            # Web interface
        
        # Documentation
        "docs/technical",     # Technical documentation
        "docs/api",          # API documentation
        "docs/guides",       # User guides
        
        # Scripts
        "scripts/data",      # Data processing scripts
        "scripts/training",  # Training scripts
        "scripts/deploy"     # Deployment scripts
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def create_documentation():
    """Create focused documentation."""
    docs = {
        # Project overview
        "docs/README.md": """# Pickleball Vision LLM

## Project Overview
An AI-powered system that combines computer vision and language models to provide real-time pickleball game analysis and coaching insights.

## Key Components
- Computer Vision (YOLO, SAM, MediaPipe)
- Language Models (BLIP-2, LLaMA, Mistral)
- Real-time Analysis Pipeline
- Coaching Insights Generation

## Architecture
- Vision Processing
- Game Analysis
- Language Generation
- Real-time Optimization

## Applications
- Players
- Coaches
- Broadcasters
- Sports Analytics
""",

        # Technical documentation
        "docs/technical/README.md": """# Technical Documentation

## Vision Pipeline
- Ball Detection & Tracking
- Player Pose Analysis
- Court Detection
- Real-time Processing

## LLM Integration
- Game State Understanding
- Strategy Analysis
- Coaching Tips Generation
- Natural Language Output

## Data Pipeline
- Video Ingestion
- Feature Extraction
- Model Training
- Performance Optimization
""",

        # Source code documentation
        "src/README.md": """# Source Code Structure

## Vision Module
- YOLO-based detection
- MediaPipe pose tracking
- Court analysis
- Real-time optimization

## LLM Module
- Game state processing
- Strategy analysis
- Coaching tip generation
- Natural language output

## Fusion Module
- Vision-LLM integration
- Real-time synchronization
- Context management
- Output optimization

## API & Web Interface
- RESTful endpoints
- Real-time streaming
- Web dashboard
- User management
"""
    }
    
    for path, content in docs.items():
        ensure_directory_exists(path)
        with open(path, "w") as f:
            f.write(content)
        print(f"Created documentation: {path}")

def cleanup_old_structure():
    """Clean up old directory structure."""
    old_directories = [
        "src/pickleball_vision/core",
        "src/pickleball_vision/infrastructure",
        "src/pickleball_vision/monitoring",
        "docs/architecture",
        "docs/development",
        "docs/prompts"
    ]
    
    for directory in old_directories:
        try:
            if Path(directory).exists():
                shutil.rmtree(directory)
                print(f"Removed old directory: {directory}")
        except Exception as e:
            print(f"Error removing {directory}: {e}")

def create_project_summary():
    """Create project summary document."""
    summary = """# Pickleball Vision LLM - Project Summary

## 🎯 Project Vision
A multi-modal AI system that combines computer vision and language models to provide real-time pickleball game analysis and coaching insights.

## 💡 Key Components

### 🔍 Vision Processing
- Ball detection and tracking (YOLO)
- Player pose analysis (MediaPipe)
- Court detection and mapping
- Real-time optimization

### 🧠 Language Understanding
- Game state interpretation
- Strategy analysis
- Coaching tip generation
- Natural language output

### 🔄 Data Pipeline
- YouTube video ingestion
- Feature extraction
- Training data generation
- Performance optimization

## 📈 Technical Advantages
1. Multi-modal integration (Vision + LLM)
2. Real-time processing capability
3. Scalable data pipeline
4. Modular architecture
5. Production-ready design

## 🎯 Target Applications
- Players: Real-time feedback
- Coaches: Game analysis
- Broadcasters: Automated insights
- Analytics: Strategic patterns

## 🚀 Future Extensions
1. Reinforcement learning for tactics
2. Player skill modeling
3. Interactive simulations
4. Cross-sport adaptation

## 🛠️ Implementation
- Clean, modular architecture
- Optimized for real-time processing
- Scalable deployment options
- Comprehensive documentation

## 📊 Success Metrics
1. Detection accuracy
2. Processing speed
3. Insight quality
4. User engagement
"""
    
    with open("docs/PROJECT_SUMMARY.md", "w") as f:
        f.write(summary)
    print("Created project summary")

def main():
    """Main function to execute focused reorganization."""
    print("Starting focused project reorganization...")
    
    # Create backup
    backup_dir = f"backup/focused_backup_{Path.cwd().name}"
    if not Path(backup_dir).exists():
        shutil.copytree(".", backup_dir, ignore=shutil.ignore_patterns("backup", ".git"))
        print(f"Created backup at: {backup_dir}")
    
    # Execute reorganization steps
    create_core_structure()
    create_documentation()
    cleanup_old_structure()
    create_project_summary()
    
    print("Project reorganization completed!")

if __name__ == "__main__":
    main() 