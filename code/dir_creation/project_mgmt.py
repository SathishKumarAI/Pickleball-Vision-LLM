#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

def get_installed_packages():
    """
    Get list of currently installed packages
    """
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'freeze'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout.strip().split('\n')
    except subprocess.CalledProcessError as e:
        print(f"Error getting installed packages: {e}")
        return []

def update_requirements_files(packages):
    """
    Update requirements files based on installed packages
    """
    # Base requirements
    base_requirements = [
        'torch', 'torchvision', 'ultralytics', 'opencv-python', 
        'mlflow', 'feast', 'prometheus-client', 'pyyaml'
    ]
    
    # Dev requirements
    dev_requirements = base_requirements + [
        'pytest', 'ruff', 'mypy', 'black'
    ]
    
    # Prod requirements
    prod_requirements = base_requirements + ['gunicorn']

    def filter_packages(package_list, requirements):
        """
        Filter and version packages
        """
        filtered = []
        for req in requirements:
            matching_packages = [
                p for p in package_list 
                if p.lower().startswith(req.lower())
            ]
            if matching_packages:
                filtered.append(matching_packages[0])
        return filtered

    installed_packages = get_installed_packages()

    # Write base requirements
    base_path = Path('requirements/base.txt')
    base_path.parent.mkdir(parents=True, exist_ok=True)
    base_path.write_text('\n'.join(
        filter_packages(installed_packages, base_requirements)
    ))

    # Write dev requirements
    dev_path = Path('requirements/dev.txt')
    dev_path.write_text(
        '-r base.txt\n' + 
        '\n'.join(filter_packages(installed_packages, dev_requirements - list(set(base_requirements)))
    ))

    # Write prod requirements
    prod_path = Path('requirements/prod.txt')
    prod_path.write_text(
        '-r base.txt\n' + 
        '\n'.join(filter_packages(installed_packages, prod_requirements - list(set(base_requirements)))
    )
    )

    print("✅ Requirements files updated successfully!")

def main():
    """
    Main function to manage requirements
    """
    update_requirements_files(get_installed_packages())

if __name__ == '__main__':
    main()