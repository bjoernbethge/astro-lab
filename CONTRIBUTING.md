# Contributing to AstroLab

Thank you for your interest in contributing to AstroLab! This document provides guidelines for contributing to our Astro GNN laboratory for cosmic web exploration.

## 🚀 Quick Start

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/astro-lab.git`
3. **Setup** the development environment:
   ```bash
   cd astro-lab
   uv sync
   ```
4. **Create** a feature branch: `git checkout -b feature/amazing-feature`
5. **Make** your changes
6. **Test** your changes: `uv run pytest -v`
7. **Commit** with clear messages: `git commit -m "feat: add cosmic web clustering algorithm"`
8. **Push** to your fork: `git push origin feature/amazing-feature`
9. **Create** a Pull Request

## 🧠 Development Focus Areas

### **Cosmic Web Analysis**
- Multi-scale clustering algorithms
- Filament detection methods (MST, Morse theory, Hessian)
- Structure identification across astronomical scales
- Interactive 3D visualization improvements

### **Graph Neural Networks**
- New GNN architectures for spatial data
- Attention mechanisms for astronomical graphs
- Multi-scale graph learning
- Temporal GNNs for variable objects

### **Data Processing**
- New survey integrations
- Improved tensor operations
- Performance optimizations
- Data validation and quality checks

### **Visualization**
- New visualization backends
- Interactive 3D features
- Scientific color mapping
- Real-time data streaming

## 📋 Code Style

### **Python Code**
- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Keep functions focused and small

### **Documentation**
- Update docstrings for new functions
- Add examples in docstrings
- Update README.md for new features
- Generate API documentation: `python docs/generate_docs.py update`

### **Testing**
- Write tests for new functionality
- Ensure all tests pass: `uv run pytest -v`
- Add integration tests for cosmic web features
- Test with real astronomical data

## 🎯 Commit Message Format

Use conventional commit messages:

```
type(scope): description

feat(cosmic_web): add MST filament detection
fix(gnn): resolve memory leak in graph convolution
docs(readme): update installation instructions
test(data): add Gaia DR3 integration tests
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## 🔬 Research Contributions

### **Astronomical Research**
- Novel clustering algorithms for cosmic web
- New filament detection methods
- Multi-scale structure analysis
- Cross-survey data integration

### **Machine Learning Research**
- Graph neural network architectures
- Attention mechanisms for spatial data
- Multi-modal learning approaches
- Transfer learning for astronomical data

### **Visualization Research**
- Scientific visualization techniques
- Interactive 3D rendering
- Real-time data visualization
- Multi-backend rendering systems

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment**: OS, Python version, package versions
2. **Steps**: Clear steps to reproduce the issue
3. **Expected**: What you expected to happen
4. **Actual**: What actually happened
5. **Data**: Sample data or code to reproduce
6. **Logs**: Error messages and stack traces

## 💡 Feature Requests

For feature requests, please describe:

1. **Problem**: What problem does this solve?
2. **Solution**: How should it work?
3. **Use Case**: Specific astronomical use case
4. **Priority**: High/Medium/Low priority
5. **Implementation**: Any implementation ideas?

## 🤝 Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Share knowledge and expertise
- Provide constructive feedback
- Celebrate contributions and achievements

## 📚 Resources

- **[API Documentation](docs/api.md)** - Complete API reference
- **[Cosmic Web Guide](docs/cosmic_web_guide.md)** - Cosmic web analysis tutorials
- **[PyTorch Documentation](https://pytorch.org/docs/)** - Deep learning framework
- **[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)** - Graph neural networks
- **[AstroPy Documentation](https://docs.astropy.org/)** - Astronomical calculations

## 🏆 Recognition

Contributors will be recognized in:
- Project README acknowledgments
- Release notes
- Documentation credits
- Community highlights

Thank you for contributing to the future of astronomical machine learning! 🌌✨ 