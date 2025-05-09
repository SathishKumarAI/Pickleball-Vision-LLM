#!/usr/bin/env python3
"""
Smart File Organizer

This script intelligently organizes files by examining both their names and content,
copying them to appropriate directories while preserving the originals.
"""

import os
import shutil
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('organize.log'),
        logging.StreamHandler()
    ]
)

# File type signatures
FILE_SIGNATURES: Dict[str, Set[str]] = {
    'python': {'def ', 'class ', 'import ', 'from ', 'print('},
    'javascript': {'function ', 'import ', 'export ', 'console.log', 'const ', 'let '},
    'html': {'<!DOCTYPE', '<html', '<head', '<body', '<div'},
    'css': {'{', '}', 'color:', 'background:', '@media'},
    'markdown': {'# ', '## ', '- ', '* ', '```'},
    'json': {'{', '}', '":', '": '},
    'yaml': {'---', ':', '- ', 'version:', 'name:'},
    'shell': {'#!/bin/bash', '#!/bin/sh', '#!/usr/bin/env bash'},
    'docker': {'FROM ', 'RUN ', 'COPY ', 'ENV ', 'WORKDIR '},
    'config': {'.env', 'config.', 'settings.', 'conf.'},
}

# Directory mapping
DIRECTORY_MAP = {
    'python': 'src/python',
    'javascript': 'src/js',
    'html': 'src/html',
    'css': 'src/css',
    'markdown': 'docs',
    'json': 'config',
    'yaml': 'config',
    'shell': 'scripts',
    'docker': 'deployment',
    'config': 'config',
    'unknown': 'misc'
}

def classify_file(file_path: Path) -> str:
    """
    Classify a file based on its extension and content.
    Returns the target directory name.
    """
    # First check file extension
    ext = file_path.suffix.lower()
    if ext in ['.py']:
        return 'python'
    elif ext in ['.js', '.jsx', '.ts', '.tsx']:
        return 'javascript'
    elif ext in ['.html', '.htm']:
        return 'html'
    elif ext in ['.css', '.scss', '.sass']:
        return 'css'
    elif ext in ['.md', '.markdown']:
        return 'markdown'
    elif ext in ['.json']:
        return 'json'
    elif ext in ['.yml', '.yaml']:
        return 'yaml'
    elif ext in ['.sh', '.bash']:
        return 'shell'
    elif ext in ['.dockerfile', ''] and file_path.name.lower() == 'dockerfile':
        return 'docker'
    
    # If extension doesn't give a clear answer, check content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(1000)  # Read first 1000 chars
            
            for file_type, signatures in FILE_SIGNATURES.items():
                if any(sig in content for sig in signatures):
                    return file_type
    except (UnicodeDecodeError, IOError):
        # Binary file or unreadable
        pass
    
    return 'unknown'

def get_unique_destination(source_path: Path, target_dir: Path) -> Path:
    """
    Generate a unique destination path, adding timestamp if needed.
    """
    dest_path = target_dir / source_path.name
    if not dest_path.exists():
        return dest_path
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name = source_path.stem
    ext = source_path.suffix
    return target_dir / f"{name}_{timestamp}{ext}"

def organize_files(source_dir: Path, target_dir: Path, dry_run: bool = False) -> None:
    """
    Organize files from source directory to target directory.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    if not source_dir.exists():
        logging.error(f"Source directory {source_dir} does not exist!")
        return
    
    # Create target directory if it doesn't exist
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all files
    for file_path in source_dir.rglob('*'):
        if file_path.is_file():
            # Skip hidden files and directories
            if any(part.startswith('.') for part in file_path.parts):
                continue
                
            # Classify the file
            file_type = classify_file(file_path)
            target_subdir = target_dir / DIRECTORY_MAP[file_type]
            
            # Get unique destination path
            dest_path = get_unique_destination(file_path, target_subdir)
            
            # Log the action
            logging.info(f"Classified {file_path} as {file_type}")
            logging.info(f"Would copy to: {dest_path}")
            
            if not dry_run:
                # Create target directory
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                # Copy the file
                shutil.copy2(file_path, dest_path)
                logging.info(f"Copied {file_path} to {dest_path}")

def main():
    parser = argparse.ArgumentParser(description='Intelligently organize files based on content and type.')
    parser.add_argument('--source', default='backup', help='Source directory containing files to organize')
    parser.add_argument('--target', default='organized', help='Target directory for organized files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    args = parser.parse_args()
    
    organize_files(args.source, args.target, args.dry_run)

if __name__ == '__main__':
    main() 