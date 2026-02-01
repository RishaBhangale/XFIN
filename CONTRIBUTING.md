# Contributing to XFIN

Thank you for your interest in contributing to XFIN! This document provides guidelines for contributors.

## 🏁 Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/XFIN.git
   cd XFIN
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. **Run tests** to ensure everything works:
   ```bash
   pytest
   ```

## 📝 Code Style

- Follow **PEP 8** style guidelines
- Use **type hints** for function signatures
- Write **docstrings** for all public functions
- Keep lines under 100 characters

## 🧪 Testing

- Add tests for new features in the `tests/` directory
- Run the full test suite before submitting:
  ```bash
  pytest --tb=short
  ```
- Aim for 80%+ code coverage

## 📤 Submitting Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes and commit:
   ```bash
   git commit -m "Add: brief description of your changes"
   ```
3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
4. Open a **Pull Request** on GitHub

## 🐛 Reporting Issues

- Use the GitHub issue tracker
- Include Python version and OS
- Provide minimal reproducible examples
- Check existing issues before creating new ones

## 📋 Areas for Contribution

- 🧪 New stress testing scenarios
- 🌍 Additional ESG data sources
- 📊 Enhanced visualizations
- 🌐 Documentation improvements
- 🐛 Bug fixes

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.
