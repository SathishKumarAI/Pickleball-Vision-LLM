# Pickleball Vision Project

## Project Structure

### Source Code (`src/pickleball_vision/`)
- **core/**: Core components and utilities
  - `config/`: Configuration management
  - `utils/`: Utility functions
  - `database/`: Database components
- **vision/**: Vision processing components
  - `detection/`: Object detection
  - `tracking/`: Object tracking
  - `preprocessing/`: Image preprocessing
- **ml/**: Machine learning components
  - `training/`: Model training scripts
  - `experiments/`: Experiment tracking
- **api/**: API endpoints and services
- **frontend/**: Frontend components
- **infrastructure/**: Infrastructure components
  - `monitoring/`: Monitoring setup
  - `nginx/`: Nginx configuration
  - `grafana/`: Grafana dashboards

### Documentation (`docs/`)
- **architecture/**: System architecture documentation
  - `system/`: System overview
  - `components/`: Component design
  - `data_flow/`: Data flow diagrams
- **api/**: API documentation
  - `endpoints/`: API endpoints
  - `models/`: Data models
  - `examples/`: Usage examples
- **guides/**: User guides
  - `installation/`: Installation guides
  - `usage/`: Usage guides
  - `troubleshooting/`: Troubleshooting guides
- **development/**: Development documentation
  - `setup/`: Development setup
  - `contributing/`: Contribution guidelines
  - `testing/`: Testing guidelines
- **prompts/**: LLM prompts
  - `vision/`: Vision-related prompts
  - `analysis/`: Analysis prompts
  - `common/`: Common prompts

### Scripts (`scripts/`)
- **setup/**: Setup scripts
- **monitoring/**: Monitoring scripts
- **deployment/**: Deployment scripts
- **utils/**: Utility scripts

### Data (`data/`)
- **raw/**: Raw data
- **processed/**: Processed data
- **test/**: Test data
- **models/**: Model files

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up the environment:
   ```bash
   scripts/setup/setup_env.sh
   ```

3. Run tests:
   ```bash
   pytest tests/
   ```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
