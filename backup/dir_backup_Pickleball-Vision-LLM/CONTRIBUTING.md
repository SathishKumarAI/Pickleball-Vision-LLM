# Contributing to Pickleball Vision

## Development Setup

### Prerequisites
- Python 3.8 or higher
- Docker and Docker Compose
- Git
- Make (optional, but recommended)

### Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pickleball-vision.git
cd pickleball-vision
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
pip install -r requirements-dev.txt
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

### Using Docker for Development

1. Build and start services:
```bash
docker-compose up --build
```

2. Run tests in container:
```bash
docker-compose exec app pytest
```

## Project Structure

```
src/pickleball_vision/
├── core/           # Core business logic
├── features/       # Feature modules
│   ├── detection/
│   ├── analysis/
│   └── visualization/
├── infrastructure/ # Infrastructure components
├── interfaces/     # API and UI interfaces
└── shared/        # Shared utilities
```

## Code Style

- We use Black for code formatting
- Type hints are required for all new code
- Documentation strings should follow Google style
- Maximum line length is 88 characters

## Testing

- Write tests for all new features
- Maintain minimum 80% code coverage
- Run tests locally before pushing:
```bash
pytest tests/
```

## Commit Guidelines

Format: `<type>(<scope>): <subject>`

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Formatting
- refactor: Code restructuring
- test: Tests
- chore: Maintenance

Example: `feat(detection): add ball tracking algorithm`

## Pull Request Process

1. Create feature branch from main
2. Write clear commit messages
3. Update documentation
4. Add tests
5. Ensure CI passes
6. Request review

## Release Process

1. Update version in setup.py
2. Update CHANGELOG.md
3. Create release PR
4. Tag release after merge 