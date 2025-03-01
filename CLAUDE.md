# CLAUDE.md - Guidelines for Paleoclimate Reconstruction Project

## Run Commands
- Run main program: `python main.py`
- Run quick demo: `python quick_start.py`
- Run model comparison: `python model_comparison.py`
- Generate synthetic data: `python synthetic_data.py`

## Code Style Guidelines
- **Imports**: Standard library first, third-party next, local modules last
- **Naming**: snake_case for variables/functions, CamelCase for classes
- **Documentation**: Google-style docstrings with Parameters/Returns sections
- **Formatting**: 4-space indentation, blank lines between logical sections
- **Project Structure**: Flat module structure with direct imports
- **Error Handling**: Input validation before model fitting/prediction
- **Scientific Conventions**: Use NumPy for array operations, Matplotlib for visualization
- **File Structure**: Each module contains related functionality (models, visualization, etc.)
- **Comments**: Include inline comments for complex algorithms

## Data Management
- Output directory for results: `data/results/`
- Set random seed (np.random.seed(42)) for reproducibility