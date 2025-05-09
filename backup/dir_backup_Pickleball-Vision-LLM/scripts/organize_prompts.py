import os
import shutil
from pathlib import Path

def move_prompt_files():
    """Move prompt files to their respective directories."""
    prompt_moves = {
        # Vision prompts
        "docs/prompts/frame_analysis_prompts.md": "docs/prompts/vision/frame_analysis.md",
        "docs/prompts/ball_detection_prompts.md": "docs/prompts/vision/ball_detection.md",
        
        # Analysis prompts
        "docs/prompts/metrics_monitoring_prompts.md": "docs/prompts/analysis/metrics_monitoring.md",
        
        # Common prompts
        "docs/prompts/common_prompts.md": "docs/prompts/common/main.md",
        "docs/prompts/quick_reference.md": "docs/prompts/common/quick_reference.md"
    }
    
    for src, dst in prompt_moves.items():
        try:
            if Path(src).exists():
                # Create parent directory if it doesn't exist
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(src, dst)
                print(f"Moved {src} to {dst}")
        except Exception as e:
            print(f"Error moving {src} to {dst}: {e}")

def create_index_files():
    """Create index files for each documentation section."""
    # Main prompts index
    prompts_index = """# Prompts Documentation

## Vision Prompts
- [Frame Analysis](vision/frame_analysis.md) - Prompts for analyzing frame quality and content
- [Ball Detection](vision/ball_detection.md) - Prompts for detecting and tracking the pickleball

## Analysis Prompts
- [Metrics Monitoring](analysis/metrics_monitoring.md) - Prompts for monitoring system metrics and performance

## Common Prompts
- [Quick Reference](common/quick_reference.md) - Quick reference guide for common prompts
- [Main Prompts](common/main.md) - Collection of frequently used prompts

## Usage
1. Select the appropriate prompt category based on your task
2. Use the provided prompts as templates
3. Customize the prompts according to your specific needs
4. Follow the best practices outlined in each prompt file
"""
    
    # Vision prompts index
    vision_index = """# Vision Prompts

## Frame Analysis
- Frame quality assessment
- Motion detection
- Frame preprocessing

## Ball Detection
- Initial ball detection
- Ball tracking
- Multiple ball handling

## Usage Guidelines
1. Start with frame analysis for video preprocessing
2. Use ball detection prompts for object detection
3. Follow the provided input/output formats
4. Monitor detection confidence scores
"""
    
    # Analysis prompts index
    analysis_index = """# Analysis Prompts

## Metrics Monitoring
- Performance metrics collection
- Quality metrics analysis
- Alert generation

## Usage Guidelines
1. Use metrics monitoring prompts for system analysis
2. Follow the specified metric collection intervals
3. Monitor alert thresholds
4. Document any custom metrics
"""
    
    # Common prompts index
    common_index = """# Common Prompts

## Quick Reference
- Code organization prompts
- Documentation prompts
- Testing prompts
- Configuration prompts
- Error handling prompts
- Performance prompts

## Main Prompts
- Detailed prompt collections
- Usage examples
- Best practices

## Usage Guidelines
1. Start with the quick reference for common tasks
2. Use main prompts for detailed implementations
3. Follow the provided templates
4. Customize as needed
"""
    
    # Write index files
    index_files = {
        "docs/prompts/README.md": prompts_index,
        "docs/prompts/vision/README.md": vision_index,
        "docs/prompts/analysis/README.md": analysis_index,
        "docs/prompts/common/README.md": common_index
    }
    
    for path, content in index_files.items():
        with open(path, "w") as f:
            f.write(content)
        print(f"Created index file: {path}")

def update_cross_references():
    """Update cross-references in documentation files."""
    # Update main README.md
    readme_path = "README.md"
    if Path(readme_path).exists():
        with open(readme_path, "r") as f:
            content = f.read()
        
        # Update prompts section
        prompts_section = """### Prompts (`docs/prompts/`)
- **Vision/**: Vision-related prompts
  - [Frame Analysis](docs/prompts/vision/frame_analysis.md)
  - [Ball Detection](docs/prompts/vision/ball_detection.md)
- **Analysis/**: Analysis prompts
  - [Metrics Monitoring](docs/prompts/analysis/metrics_monitoring.md)
- **Common/**: Common prompts
  - [Quick Reference](docs/prompts/common/quick_reference.md)
  - [Main Prompts](docs/prompts/common/main.md)
"""
        
        # Replace the prompts section in the README
        if "### Prompts" in content:
            content = content.split("### Prompts")[0] + prompts_section + content.split("### Scripts")[1]
        
        with open(readme_path, "w") as f:
            f.write(content)
        print(f"Updated cross-references in {readme_path}")

def main():
    """Main function to execute prompt organization."""
    print("Starting prompt organization...")
    
    # Create backup
    backup_dir = f"backup/prompts_backup_{Path.cwd().name}"
    if not Path(backup_dir).exists():
        shutil.copytree(".", backup_dir, ignore=shutil.ignore_patterns("backup", ".git"))
        print(f"Created backup at: {backup_dir}")
    
    # Execute organization steps
    move_prompt_files()
    create_index_files()
    update_cross_references()
    
    print("Prompt organization completed!")

if __name__ == "__main__":
    main() 