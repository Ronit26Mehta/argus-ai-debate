# Contributing to ARGUS

Thank you for your interest in contributing to ARGUS! This document provides guidelines and instructions for contributing.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

---

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please:

- Be respectful and considerate in all interactions
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Accept responsibility for mistakes and learn from them

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- A GitHub account

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR-USERNAME/argus-ai-debate.git
cd argus-ai-debate
```

3. Add the upstream remote:

```bash
git remote add upstream https://github.com/Ronit26Mehta/argus-ai-debate.git
```

---

## Development Setup

### Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### Install Dependencies

```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Verify Setup

```bash
# Run tests
pytest

# Check code style
ruff check argus/
black --check argus/

# Type checking
mypy argus/
```

---

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include:
   - Python version
   - ARGUS version
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages/tracebacks

### Suggesting Features

1. Check existing issues and discussions
2. Open a feature request issue
3. Describe the use case and expected behavior
4. Be open to feedback and iteration

### Contributing Code

1. Find an issue to work on (or create one)
2. Comment on the issue to indicate you're working on it
3. Create a feature branch
4. Make your changes
5. Submit a pull request

---

## Pull Request Process

### Before Submitting

1. **Sync with upstream**:
```bash
git fetch upstream
git rebase upstream/main
```

2. **Create a feature branch**:
```bash
git checkout -b feature/your-feature-name
```

3. **Make your changes** following the coding standards

4. **Run tests**:
```bash
pytest
```

5. **Check code style**:
```bash
ruff check argus/
black argus/
```

6. **Commit with clear messages**:
```bash
git commit -m "Add feature: description of change"
```

### Submitting

1. Push to your fork:
```bash
git push origin feature/your-feature-name
```

2. Open a Pull Request on GitHub

3. Fill out the PR template with:
   - Description of changes
   - Related issue numbers
   - Testing performed
   - Breaking changes (if any)

### Review Process

- Maintainers will review your PR
- Address feedback and make requested changes
- Once approved, a maintainer will merge your PR

---

## Coding Standards

### Style Guide

- Follow PEP 8 guidelines
- Use Black for formatting (line length: 100)
- Use Ruff for linting
- Use type hints for all functions

### Code Formatting

```bash
# Format code
black argus/ tests/

# Sort imports
ruff check --fix argus/
```

### Naming Conventions

- **Classes**: PascalCase (e.g., `BayesianUpdater`)
- **Functions/Methods**: snake_case (e.g., `compute_posterior`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_ITERATIONS`)
- **Private members**: prefix with underscore (e.g., `_internal_method`)

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include type hints in signatures

Example:
```python
def compute_posterior(
    graph: CDAG,
    proposition_id: str,
    config: Optional[PropagationConfig] = None,
) -> float:
    """
    Compute Bayesian posterior for a proposition.
    
    Args:
        graph: The C-DAG containing the proposition
        proposition_id: ID of the proposition to evaluate
        config: Optional propagation configuration
        
    Returns:
        Posterior probability between 0 and 1
        
    Raises:
        ValueError: If proposition_id not found in graph
    """
```

---

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=argus --cov-report=html

# Run specific test file
pytest tests/unit/test_cdag.py

# Run specific test
pytest tests/unit/test_cdag.py::TestProposition::test_creation

# Run with verbose output
pytest -v
```

### Writing Tests

- Place tests in `tests/unit/` or `tests/integration/`
- Name test files as `test_<module>.py`
- Name test functions as `test_<description>`
- Use pytest fixtures from `conftest.py`
- Aim for high coverage of new code

Example:
```python
import pytest
from argus.cdag import Proposition

class TestProposition:
    def test_creation(self):
        prop = Proposition(text="Test", prior=0.5)
        assert prop.prior == 0.5
        assert prop.posterior == 0.5
    
    def test_invalid_prior(self):
        with pytest.raises(ValidationError):
            Proposition(text="Test", prior=1.5)
```

### Test Categories

- **Unit tests**: Test individual functions/classes in isolation
- **Integration tests**: Test component interactions
- Use `@pytest.mark.slow` for slow tests
- Use `@pytest.mark.integration` for integration tests

---

## Documentation

### Updating Documentation

- Update docstrings when changing function behavior
- Update README.md for user-facing changes
- Add examples for new features

### Building Documentation

```bash
# If using Sphinx (future)
cd docs
make html
```

---

## Project Structure

```
argus/
├── argus/              # Main package
│   ├── agents/         # Agent implementations
│   ├── cdag/           # Debate graph
│   ├── core/           # Core infrastructure
│   ├── decision/       # Decision layer
│   ├── knowledge/      # Knowledge layer
│   ├── provenance/     # Provenance tracking
│   └── retrieval/      # Retrieval layer
├── tests/              # Test suite
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── README.md
├── CONTRIBUTING.md
├── LICENSE
└── pyproject.toml
```

---

## Questions?

- Open a GitHub issue for questions
- Tag maintainers for urgent matters

Thank you for contributing to ARGUS!
