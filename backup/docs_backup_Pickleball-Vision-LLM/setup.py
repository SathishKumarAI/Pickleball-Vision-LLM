from setuptools import setup, find_packages
import io

setup(
    name="pickleball-vision",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Core dependencies
        "numpy>=1.24.2,<2.0.0",
        "pandas>=1.5.3,<2.0.0",
        "torch>=2.0.0,<3.0.0",
        "cupy-cuda12x>=12.0.0",  # For GPU acceleration
        "opencv-python>=4.7.0,<5.0.0",
        "ultralytics>=8.0.0,<9.0.0",
        
        # Web and visualization
        "streamlit>=1.24.0,<2.0.0",
        "plotly>=5.13.0,<6.0.0",
        
        # ML and data processing
        "scikit-learn>=1.0.0,<2.0.0",
        "scipy>=1.10.0,<2.0.0",
        
        # Utilities
        "click>=8.1.3,<9.0.0",
        "pyyaml>=6.0,<7.0.0",
        "loguru>=0.7.0,<1.0.0",
        "mlflow>=2.8.0,<3.0.0",
        "asyncio>=3.4.3,<4.0.0",
        "colorama>=0.4.6,<1.0.0",  # For colored console output
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0,<8.0.0",
            "pytest-asyncio>=0.21.0,<1.0.0",
            "black>=22.0.0,<23.0.0",
            "isort>=5.0.0,<6.0.0",
            "flake8>=4.0.0,<5.0.0",
            "mypy>=0.900,<1.0.0",
            "pre-commit>=3.0.0,<4.0.0",
        ],
        "gpu": [
            "cupy-cuda12x>=12.0.0",
            "torch>=2.0.0,<3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pickleball-vision=pickleball_vision.cli:cli",
        ],
    },
    python_requires=">=3.8,<3.12",
    author="Your Name",
    author_email="your.email@example.com",
    description="A computer vision system for pickleball analysis",
    long_description=io.open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pickleball-vision",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Computer Vision",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
) 