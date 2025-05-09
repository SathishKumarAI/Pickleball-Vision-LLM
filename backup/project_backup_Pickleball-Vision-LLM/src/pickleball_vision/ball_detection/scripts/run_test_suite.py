"""
Test suite for data collection pipeline.

This script runs multiple test cases with different configurations
to evaluate the data collection pipeline's performance.
"""

import yaml
from pathlib import Path
import logging
from test_collection import test_data_collection
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_configs():
    """Create different test configurations."""
    base_config = {
        'video_dir': "data/raw/videos",
        'output_dir': "data/processed",
        'min_frames_per_video': 100,
        'max_frames_per_video': 1000,
        'frame_interval': 1,
        'brightness_range': {'min': 30, 'max': 225},
        'contrast_threshold': 20,
        'blur_threshold': 100,
        'motion_threshold': 0.1,
        'optical_flow': {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2
        },
        'contrast_enhancement': {
            'clip_limit': 3.0,
            'tile_size': 8
        },
        'denoising': {
            'h': 10,
            'template_size': 7,
            'search_size': 21
        }
    }
    
    # Create test configurations
    configs = {
        'default': base_config.copy(),
        'high_quality': base_config.copy(),
        'high_motion': base_config.copy(),
        'balanced': base_config.copy()
    }
    
    # Modify configurations
    configs['high_quality'].update({
        'brightness_range': {'min': 50, 'max': 200},
        'contrast_threshold': 30,
        'blur_threshold': 150
    })
    
    configs['high_motion'].update({
        'motion_threshold': 0.2,
        'optical_flow': {
            'pyr_scale': 0.5,
            'levels': 4,
            'winsize': 20,
            'iterations': 4,
            'poly_n': 7,
            'poly_sigma': 1.5
        }
    })
    
    configs['balanced'].update({
        'brightness_range': {'min': 40, 'max': 210},
        'contrast_threshold': 25,
        'blur_threshold': 120,
        'motion_threshold': 0.15
    })
    
    return configs

def run_test_suite(video_path: str):
    """
    Run test suite with different configurations.
    
    Args:
        video_path: Path to test video
    """
    logger.info("Starting test suite...")
    
    # Create test configurations
    configs = create_test_configs()
    
    # Create test directory
    results_dir = Path("test_results")
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir()
    
    # Run tests
    results = {}
    for config_name, config in configs.items():
        logger.info(f"\nRunning test case: {config_name}")
        
        # Save configuration
        config_path = results_dir / f"{config_name}_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
        # Run test
        try:
            test_results = test_data_collection(video_path, str(config_path))
            results[config_name] = test_results
        except Exception as e:
            logger.error(f"Test case {config_name} failed: {e}")
            results[config_name] = {'error': str(e)}
    
    # Save combined results
    with open(results_dir / "test_suite_results.yaml", 'w') as f:
        yaml.dump(results, f)
        
    # Print summary
    logger.info("\nTest Suite Summary:")
    for config_name, result in results.items():
        if 'error' in result:
            logger.info(f"\n{config_name}: Failed - {result['error']}")
        else:
            stats = result['statistics']
            logger.info(f"\n{config_name}:")
            logger.info(f"  Total frames: {stats['total_frames']}")
            logger.info(f"  Avg quality: {stats['quality']['mean']:.2f}")
            logger.info(f"  Avg motion: {stats['motion']['mean']:.2f}")
    
    logger.info(f"\nTest suite results saved to {results_dir}")

def main():
    """Main function to run test suite."""
    # Get video path from user
    video_path = input("Enter path to test video: ")
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        return
        
    # Run test suite
    run_test_suite(video_path)

if __name__ == "__main__":
    main() 