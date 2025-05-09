import os
import shutil
from pathlib import Path

def ensure_directory_exists(path):
    """Ensure directory exists before writing file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def create_directory_indexes():
    """Create index files for all main directories."""
    indexes = {
        # Architecture documentation
        "docs/architecture/README.md": """# Architecture Documentation

## System Overview
- [System Architecture](system/overview.md) - High-level system design and components
- [Data Flow](data_flow/main.md) - Data flow diagrams and processing pipelines
- [Component Design](components/design.md) - Detailed component specifications

## Key Components
- Vision Processing Pipeline
- Ball Detection System
- Tracking System
- API Services
- Frontend Interface

## Design Principles
- Modular Architecture
- Scalable Components
- Real-time Processing
- Fault Tolerance
""",

        # API documentation
        "docs/api/README.md": """# API Documentation

## Endpoints
- [API Reference](endpoints/main.md) - Complete API endpoint documentation
- [Authentication](endpoints/auth.md) - Authentication and authorization
- [Rate Limiting](endpoints/rate_limits.md) - API usage limits

## Models
- [Data Models](models/main.md) - Request/response data structures
- [Validation](models/validation.md) - Input validation rules

## Examples
- [Usage Examples](examples/main.md) - Common API usage patterns
- [Error Handling](examples/errors.md) - Error handling and troubleshooting
""",

        # Guides
        "docs/guides/README.md": """# User Guides

## Installation
- [Setup Guide](installation/main.md) - Installation and setup instructions
- [Environment Setup](installation/env.md) - Environment configuration
- [Dependencies](installation/deps.md) - Required dependencies

## Usage
- [Getting Started](usage/main.md) - Quick start guide
- [Basic Usage](usage/basic.md) - Basic usage patterns
- [Advanced Features](usage/advanced.md) - Advanced features and configurations

## Troubleshooting
- [Common Issues](troubleshooting/main.md) - Common problems and solutions
- [Debug Guide](troubleshooting/debug.md) - Debugging techniques
- [Performance](troubleshooting/performance.md) - Performance optimization
""",

        # Development
        "docs/development/README.md": """# Development Guide

## Setup
- [Development Environment](setup/main.md) - Setting up development environment
- [Code Style](setup/style.md) - Coding standards and style guide
- [Tools](setup/tools.md) - Development tools and utilities

## Contributing
- [Contribution Guide](contributing/main.md) - How to contribute
- [Pull Requests](contributing/pr.md) - PR process and guidelines
- [Code Review](contributing/review.md) - Code review process

## Testing
- [Test Guide](testing/main.md) - Testing guidelines
- [Unit Tests](testing/unit.md) - Unit testing
- [Integration Tests](testing/integration.md) - Integration testing
""",

        # Source code
        "src/pickleball_vision/README.md": """# Pickleball Vision Source Code

## Core Components
- [Configuration](core/config/README.md) - Configuration management
- [Utilities](core/utils/README.md) - Utility functions
- [Database](core/database/README.md) - Database components

## Vision Components
- [Detection](vision/detection/README.md) - Object detection
- [Tracking](vision/tracking/README.md) - Object tracking
- [Preprocessing](vision/preprocessing/README.md) - Image preprocessing

## ML Components
- [Training](ml/training/README.md) - Model training
- [Experiments](ml/experiments/README.md) - Experiment tracking

## Application Components
- [API](api/README.md) - API endpoints
- [Frontend](frontend/README.md) - Frontend interface
- [Services](services/README.md) - Backend services

## Infrastructure
- [Monitoring](infrastructure/monitoring/README.md) - System monitoring
- [Nginx](infrastructure/nginx/README.md) - Web server configuration
- [Grafana](infrastructure/grafana/README.md) - Metrics visualization
""",

        # Scripts
        "scripts/README.md": """# Scripts

## Setup Scripts
- [Environment Setup](setup/README.md) - Environment configuration
- [Dependencies](setup/deps.md) - Dependency management
- [Initialization](setup/init.md) - System initialization

## Monitoring Scripts
- [Metrics Collection](monitoring/README.md) - Metrics gathering
- [Alerting](monitoring/alerts.md) - Alert configuration
- [Logging](monitoring/logs.md) - Log management

## Deployment Scripts
- [Deployment](deployment/README.md) - Deployment procedures
- [Configuration](deployment/config.md) - Deployment configuration
- [Rollback](deployment/rollback.md) - Rollback procedures

## Utility Scripts
- [Data Processing](utils/README.md) - Data processing utilities
- [Testing](utils/testing.md) - Test utilities
- [Maintenance](utils/maintenance.md) - Maintenance scripts
""",

        # Data
        "data/README.md": """# Data Directory

## Raw Data
- [Video Data](raw/README.md) - Raw video files
- [Annotations](raw/annotations.md) - Data annotations
- [Metadata](raw/metadata.md) - Data metadata

## Processed Data
- [Preprocessed](processed/README.md) - Preprocessed data
- [Features](processed/features.md) - Extracted features
- [Labels](processed/labels.md) - Processed labels

## Test Data
- [Test Sets](test/README.md) - Test datasets
- [Validation](test/validation.md) - Validation sets
- [Evaluation](test/evaluation.md) - Evaluation data

## Models
- [Trained Models](models/README.md) - Saved model files
- [Checkpoints](models/checkpoints.md) - Model checkpoints
- [Configurations](models/configs.md) - Model configurations
"""
    }
    
    for path, content in indexes.items():
        ensure_directory_exists(path)
        with open(path, "w") as f:
            f.write(content)
        print(f"Created index file: {path}")

def create_subdirectory_indexes():
    """Create index files for subdirectories."""
    subdir_indexes = {
        # Core components
        "src/pickleball_vision/core/config/README.md": """# Configuration Management

## Configuration Files
- Main configuration
- Environment variables
- Feature flags

## Usage
- Loading configurations
- Updating settings
- Validation
""",

        "src/pickleball_vision/core/utils/README.md": """# Utility Functions

## Common Utilities
- Logging
- Error handling
- File operations
- Data processing

## Usage
- Import utilities
- Common patterns
- Best practices
""",

        # Vision components
        "src/pickleball_vision/vision/detection/README.md": """# Object Detection

## Components
- Ball detection
- Player detection
- Court detection

## Usage
- Detection pipeline
- Model integration
- Performance tuning
""",

        "src/pickleball_vision/vision/tracking/README.md": """# Object Tracking

## Components
- Ball tracking
- Player tracking
- Motion analysis

## Usage
- Tracking pipeline
- State management
- Performance optimization
""",

        # ML components
        "src/pickleball_vision/ml/training/README.md": """# Model Training

## Training Pipeline
- Data preparation
- Model training
- Evaluation
- Deployment

## Usage
- Training scripts
- Configuration
- Monitoring
""",

        # Infrastructure
        "src/pickleball_vision/infrastructure/monitoring/README.md": """# System Monitoring

## Components
- Metrics collection
- Alerting
- Logging
- Visualization

## Usage
- Setup monitoring
- Configure alerts
- View metrics
"""
    }
    
    for path, content in subdir_indexes.items():
        ensure_directory_exists(path)
        with open(path, "w") as f:
            f.write(content)
        print(f"Created subdirectory index file: {path}")

def cleanup_empty_dirs():
    """Remove empty directories."""
    directories_to_remove = [
        "docs/empty",
        "docs/temp",
        "docs/draft",
        "src/pickleball_vision/empty",
        "src/pickleball_vision/temp",
        "src/pickleball_vision/draft",
        "tests/empty",
        "tests/temp",
        "scripts/empty",
        "scripts/temp"
    ]
    
    for directory in directories_to_remove:
        try:
            if Path(directory).exists():
                if not any(Path(directory).iterdir()):
                    shutil.rmtree(directory)
                    print(f"Removed empty directory: {directory}")
        except Exception as e:
            print(f"Error removing {directory}: {e}")

def main():
    """Main function to execute directory organization."""
    print("Starting directory organization...")
    
    # Create backup
    backup_dir = f"backup/dir_backup_{Path.cwd().name}"
    if not Path(backup_dir).exists():
        shutil.copytree(".", backup_dir, ignore=shutil.ignore_patterns("backup", ".git"))
        print(f"Created backup at: {backup_dir}")
    
    # Execute organization steps
    create_directory_indexes()
    create_subdirectory_indexes()
    cleanup_empty_dirs()
    
    print("Directory organization completed!")

if __name__ == "__main__":
    main() 