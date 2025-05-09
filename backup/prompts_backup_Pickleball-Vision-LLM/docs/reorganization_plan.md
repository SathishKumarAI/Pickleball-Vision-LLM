# Project Reorganization Plan

## 1. Scripts Consolidation

### Current State
- `/scripts/` (root level)
- `/src/scripts/`

### Proposed Changes
1. Move all scripts to `/scripts/` with subdirectories:
   ```
   /scripts/
   ├── setup/          # Environment and setup scripts
   ├── testing/        # Test-related scripts
   ├── deployment/     # Deployment scripts
   └── utils/          # Utility scripts
   ```

## 2. Source Code Organization

### Current State
- `/src/pickleball_vision/`
- `/src/vision/`
- `/src/app/`
- `/src/frontend/`

### Proposed Changes
1. Consolidate vision-related code:
   ```
   /src/pickleball_vision/
   ├── vision/         # Core vision algorithms
   ├── detection/      # Ball detection
   ├── tracking/       # Ball tracking
   └── preprocessing/  # Frame preprocessing
   ```

2. Organize application code:
   ```
   /src/pickleball_vision/
   ├── api/           # Backend API
   ├── frontend/      # Frontend application
   └── services/      # Business logic
   ```

## 3. Documentation Structure

### Current State
- `/docs/pickleball_vision/`
- `/docs/prompts/`

### Proposed Changes
1. Reorganize documentation:
   ```
   /docs/
   ├── api/           # API documentation
   ├── guides/        # User guides
   ├── development/   # Development docs
   ├── prompts/       # AI prompts
   └── architecture/  # System architecture
   ```

## 4. Data and Models

### Current State
- `/src/data/`
- `/src/models/`

### Proposed Changes
1. Standardize data organization:
   ```
   /data/
   ├── raw/          # Raw video data
   ├── processed/    # Processed frames
   ├── test/         # Test data
   └── models/       # Trained models
   ```

## Implementation Steps

1. **Scripts Consolidation**
   - Move all scripts to root `/scripts/`
   - Update references in documentation
   - Update Makefile targets

2. **Source Code Reorganization**
   - Merge vision-related code
   - Consolidate application code
   - Update import statements
   - Update documentation

3. **Documentation Restructuring**
   - Move files to new structure
   - Update cross-references
   - Update navigation

4. **Data Organization**
   - Move data to root `/data/`
   - Update configuration files
   - Update documentation

## Benefits

1. **Improved Organization**
   - Clear separation of concerns
   - Reduced duplication
   - Better maintainability

2. **Easier Navigation**
   - Consistent structure
   - Clear documentation
   - Better discoverability

3. **Better Development Experience**
   - Standardized locations
   - Clear dependencies
   - Simplified workflows

## Next Steps

1. Create backup of current structure
2. Implement changes incrementally
3. Update all references
4. Test all functionality
5. Update documentation 