import os
import shutil
from pathlib import Path

def create_new_directories():
    """Create new directories for merged files."""
    directories = [
        # Core components
        "src/pickleball_vision/core/config",
        "src/pickleball_vision/core/utils",
        "src/pickleball_vision/core/database",
        
        # ML and Analytics
        "src/pickleball_vision/ml/training",
        "src/pickleball_vision/ml/experiments",
        "src/pickleball_vision/analytics",
        
        # Infrastructure
        "src/pickleball_vision/infrastructure/monitoring",
        "src/pickleball_vision/infrastructure/nginx",
        "src/pickleball_vision/infrastructure/grafana",
        
        # Scripts
        "scripts/setup",
        "scripts/monitoring",
        "scripts/deployment",
        "scripts/utils"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def merge_files():
    """Merge and organize files based on relevance."""
    # Core configuration and utilities
    core_files = {
        "src/pickleball_vision/config/config.py": "src/pickleball_vision/core/config/config.py",
        "src/pickleball_vision/utils/logger.py": "src/pickleball_vision/core/utils/logger.py",
        "src/pickleball_vision/utils/colors.py": "src/pickleball_vision/core/utils/colors.py",
        "src/pickleball_vision/utils/cache_manager.py": "src/pickleball_vision/core/utils/cache_manager.py",
        "src/pickleball_vision/utils/analyzer.py": "src/pickleball_vision/core/utils/analyzer.py",
        "src/pickleball_vision/database/vector_store.py": "src/pickleball_vision/core/database/vector_store.py"
    }
    
    # ML and Analytics
    ml_files = {
        "src/scripts/scripts/train_model.py": "src/pickleball_vision/ml/training/train_model.py",
        "src/scripts/scripts/run_mlflow_experiment.py": "src/pickleball_vision/ml/experiments/run_mlflow_experiment.py",
        "src/pickleball_vision/analytics/gpu_analyzer.py": "src/pickleball_vision/analytics/gpu_analyzer.py",
        "src/pickleball_vision/analytics/ml_analyzer.py": "src/pickleball_vision/analytics/ml_analyzer.py",
        "src/pickleball_vision/analytics/stream_analyzer.py": "src/pickleball_vision/analytics/stream_analyzer.py"
    }
    
    # Infrastructure and Monitoring
    infra_files = {
        "prometheus.yml": "src/pickleball_vision/infrastructure/monitoring/prometheus.yml",
        "alert_rules.yml": "src/pickleball_vision/infrastructure/monitoring/alert_rules.yml",
        "alertmanager.yml": "src/pickleball_vision/infrastructure/monitoring/alertmanager.yml",
        "nginx/conf.d/default.conf": "src/pickleball_vision/infrastructure/nginx/default.conf",
        "grafana/provisioning/dashboards/pickleball-vision.json": "src/pickleball_vision/infrastructure/grafana/dashboards/pickleball-vision.json",
        "grafana/provisioning/datasources/prometheus.yml": "src/pickleball_vision/infrastructure/grafana/datasources/prometheus.yml",
        "grafana/provisioning/dashboards/dashboards.yml": "src/pickleball_vision/infrastructure/grafana/dashboards/dashboards.yml"
    }
    
    # Scripts
    script_files = {
        "scripts/generate-ssl.sh": "scripts/setup/generate-ssl.sh",
        "scripts/setup-monitoring.sh": "scripts/monitoring/setup-monitoring.sh",
        "scripts/process_video.py": "scripts/utils/process_video.py",
        "scripts/setup_data.py": "scripts/utils/setup_data.py"
    }
    
    # Docker and deployment
    docker_files = {
        "docker-compose.prod.yml": "deployment/docker-compose.prod.yml"
    }
    
    # Merge all file mappings
    all_files = {**core_files, **ml_files, **infra_files, **script_files, **docker_files}
    
    # Move files to their new locations
    for src, dst in all_files.items():
        try:
            if Path(src).exists():
                # Create parent directory if it doesn't exist
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(src, dst)
                print(f"Moved {src} to {dst}")
        except Exception as e:
            print(f"Error moving {src} to {dst}: {e}")

def cleanup():
    """Remove empty directories after merging."""
    directories_to_remove = [
        "src/pickleball_vision/config",
        "src/pickleball_vision/utils",
        "src/pickleball_vision/database",
        "src/scripts/scripts",
        "src/pickleball_vision/analytics",
        "nginx/conf.d",
        "grafana/provisioning"
    ]
    
    for directory in directories_to_remove:
        try:
            if Path(directory).exists():
                shutil.rmtree(directory)
                print(f"Removed directory: {directory}")
        except Exception as e:
            print(f"Error removing {directory}: {e}")

def main():
    """Main function to execute the file merging process."""
    print("Starting file merging and organization...")
    
    # Create backup
    backup_dir = f"backup/merge_backup_{Path.cwd().name}"
    if not Path(backup_dir).exists():
        shutil.copytree(".", backup_dir, ignore=shutil.ignore_patterns("backup", ".git"))
        print(f"Created backup at: {backup_dir}")
    
    # Execute merging steps
    create_new_directories()
    merge_files()
    cleanup()
    
    print("File merging and organization completed!")

if __name__ == "__main__":
    main() 