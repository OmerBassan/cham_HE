# Contributing to Chameleon

Thank you for your interest in contributing to Chameleon! This document provides guidelines for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Guidelines](#issue-guidelines)
- [Pull Request Process](#pull-request-process)

## 🤝 Code of Conduct

This project adheres to a Code of Conduct that we expect all contributors to follow. Please read and follow these guidelines to ensure a welcoming environment for everyone.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- OpenAI API access
- Basic knowledge of machine learning and API usage

### Areas for Contribution

We welcome contributions in the following areas:

- **Core Functionality**: Distortion algorithms, batch processing, analysis
- **Documentation**: Improving guides, examples, and API documentation
- **Testing**: Unit tests, integration tests, performance tests
- **Analysis**: New metrics, visualizations, statistical methods
- **Infrastructure**: CI/CD, deployment, monitoring
- **Research**: New distortion techniques, evaluation methods

## 🛠️ Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/chameleon.git
cd chameleon

# Add upstream remote
git remote add upstream https://github.com/original-owner/chameleon.git
```

### 2. Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Development Configuration

```bash
# Copy example config
cp config/config.example.yaml config/config.yaml

# Set up environment variables
export OPENAI_API_KEY="your-test-api-key"
export CHAMELEON_ENV="development"
```

### 4. Verify Setup

```bash
# Run tests to verify setup
python3 -m pytest tests/ -v

# Run basic functionality check
python3 -c "from modules.data_preparation import DistortionEngine; print('Setup OK')"
```

## 🔄 Contributing Process

### 1. Create an Issue

Before starting work, create an issue to discuss:

- Bug reports with detailed reproduction steps
- Feature requests with clear use cases
- Documentation improvements
- Performance enhancements

### 2. Branch Strategy

```bash
# Create feature branch from main
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name

# For bug fixes
git checkout -b bugfix/issue-number-description

# For documentation
git checkout -b docs/improvement-description
```

### 3. Development Workflow

```bash
# Make your changes
# Run tests frequently
python3 -m pytest tests/

# Run linting
flake8 src/ tests/
black src/ tests/

# Update documentation if needed
# Add tests for new functionality
```

### 4. Commit Guidelines

Use clear, descriptive commit messages:

```bash
# Good examples
git commit -m "feat: Add support for custom distortion functions"
git commit -m "fix: Handle edge case in batch processing"
git commit -m "docs: Update API documentation for new features"
git commit -m "test: Add unit tests for distortion engine"

# Commit message format
# type(scope): description
#
# Types: feat, fix, docs, test, refactor, style, perf, chore
```

## 📝 Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Line length: 88 characters (Black default)
# Use type hints
def process_questions(questions: List[Dict[str, Any]]) -> ProcessingResult:
    """Process questions with proper typing."""
    pass

# Docstring format (Google style)
def apply_distortion(text: str, miu: float) -> str:
    """Apply lexical distortion to text.
    
    Args:
        text: Input text to distort
        miu: Distortion intensity (0.0 to 1.0)
        
    Returns:
        Distorted text string
        
    Raises:
        ValueError: If miu is outside valid range
    """
    pass
```

### Code Organization

```python
# Import order
import os
import sys
from typing import List, Dict, Any

import pandas as pd
import numpy as np

from modules.data_preparation import DistortionEngine
from modules.gpt5_batch_processor import BatchProcessor
```

### Error Handling

```python
# Use specific exceptions
class DistortionError(Exception):
    """Raised when distortion process fails."""
    pass

# Proper error handling
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise DistortionError(f"Failed to process: {e}") from e
```

## 🧪 Testing

### Test Structure

```
tests/
├── unit/
│   ├── test_distortion_engine.py
│   ├── test_batch_processor.py
│   └── test_analysis.py
├── integration/
│   ├── test_end_to_end.py
│   └── test_api_integration.py
└── fixtures/
    ├── sample_questions.json
    └── mock_responses.json
```

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch
from modules.data_preparation import DistortionEngine

class TestDistortionEngine:
    def setup_method(self):
        self.engine = DistortionEngine()
        
    def test_apply_distortion_valid_input(self):
        """Test distortion with valid parameters."""
        text = "What is the capital of France?"
        miu = 0.5
        
        result = self.engine.apply_distortion(text, miu)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert result != text  # Should be different
        
    def test_apply_distortion_invalid_miu(self):
        """Test distortion with invalid miu values."""
        text = "Sample question"
        
        with pytest.raises(ValueError):
            self.engine.apply_distortion(text, -0.1)
            
        with pytest.raises(ValueError):
            self.engine.apply_distortion(text, 1.1)
            
    @patch('modules.data_preparation.openai_client')
    def test_batch_processing_mock(self, mock_client):
        """Test batch processing with mocked API."""
        mock_client.return_value = Mock()
        # Test implementation
```

### Running Tests

```bash
# Run all tests
python3 -m pytest

# Run specific test file
python3 -m pytest tests/unit/test_distortion_engine.py

# Run with coverage
python3 -m pytest --cov=modules tests/

# Run integration tests (requires API key)
python3 -m pytest tests/integration/ --api-key

# Performance tests
python3 -m pytest tests/performance/ --benchmark-only
```

## 📚 Documentation

### Code Documentation

- All public functions must have docstrings
- Use Google-style docstrings
- Include type hints for all function parameters and returns
- Add examples for complex functions

```python
def calculate_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Calculate accuracy between predictions and targets.
    
    Args:
        predictions: List of predicted answers (A, B, C, D)
        targets: List of correct answers (A, B, C, D)
        
    Returns:
        Accuracy as a float between 0.0 and 1.0
        
    Raises:
        ValueError: If lists have different lengths
        
    Examples:
        >>> calculate_accuracy(['A', 'B'], ['A', 'C'])
        0.5
        >>> calculate_accuracy(['A', 'A'], ['A', 'A'])
        1.0
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")
    
    correct = sum(p == t for p, t in zip(predictions, targets))
    return correct / len(predictions)
```

### API Documentation

Update API documentation when adding new features:

```python
# modules/api_docs.py
"""API Documentation for Chameleon Framework.

This module provides comprehensive documentation for all public APIs.
"""

class DistortionEngineAPI:
    """Documentation for DistortionEngine class."""
    
    def apply_distortion(self, text: str, miu: float) -> str:
        """Apply lexical distortion to input text.
        
        HTTP Equivalent:
            POST /api/v1/distortion
            {
                "text": "input text",
                "miu": 0.5
            }
        """
        pass
```

## 🐛 Issue Guidelines

### Bug Reports

Include the following information:

```markdown
## Bug Description
Brief description of the bug

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., macOS 12.0]
- Python version: [e.g., 3.9.7]
- Chameleon version: [e.g., 1.0.0]
- Dependencies: [any relevant package versions]

## Additional Context
- Error messages
- Screenshots
- Related issues
```

### Feature Requests

```markdown
## Feature Description
Clear description of the proposed feature

## Use Case
Why is this feature needed?

## Proposed Solution
How should this feature work?

## Alternatives Considered
Other solutions you've considered

## Additional Context
Any other relevant information
```

## 🔀 Pull Request Process

### Before Submitting

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Changelog is updated (if applicable)
- [ ] No merge conflicts

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or marked as such)
```

### Review Process

1. **Automated Checks**: CI/CD runs tests and linting
2. **Code Review**: Maintainers review code quality and design
3. **Testing**: Feature testing and regression testing
4. **Documentation**: Review of documentation updates
5. **Approval**: Final approval and merge

### After Merge

- Pull request author should delete their feature branch
- Monitor for any issues in the following days
- Respond to any questions or issues related to the changes

## 📊 Performance Guidelines

### Optimization Principles

- Profile before optimizing
- Optimize for common use cases
- Document performance characteristics
- Add performance tests for critical paths

### Memory Usage

```python
# Good: Generator for large datasets
def process_large_dataset(data_path: str):
    for chunk in pd.read_csv(data_path, chunksize=1000):
        yield process_chunk(chunk)

# Bad: Loading everything into memory
def process_large_dataset_bad(data_path: str):
    data = pd.read_csv(data_path)  # Could be huge
    return process_all_data(data)
```

### API Efficiency

```python
# Good: Batch processing
def process_questions_batch(questions: List[str]) -> List[str]:
    # Process multiple questions in one API call
    return batch_api_call(questions)

# Bad: Individual calls
def process_questions_individual(questions: List[str]) -> List[str]:
    results = []
    for question in questions:
        result = single_api_call(question)  # Inefficient
        results.append(result)
    return results
```

## 🔒 Security Considerations

### API Keys

- Never commit API keys to version control
- Use environment variables or secure credential storage
- Rotate keys regularly
- Use least-privilege access

### Data Handling

- Sanitize input data
- Validate all user inputs
- Handle sensitive information appropriately
- Follow data protection guidelines

## 📞 Getting Help

### Communication Channels

- **GitHub Discussions**: General questions and discussions
- **GitHub Issues**: Bug reports and feature requests
- **Email**: chameleon-dev@example.com (for sensitive issues)

### Response Times

- **Bug reports**: 48 hours for initial response
- **Feature requests**: 1 week for initial response
- **Security issues**: 24 hours for critical issues
- **Pull requests**: 72 hours for initial review

## 🎉 Recognition

Contributors will be recognized in:

- AUTHORS.md file
- GitHub contributors page
- Release notes for significant contributions
- Annual contributor highlights

Thank you for contributing to Chameleon! Your efforts help make this project better for everyone.
