# Contributing to OpenEvolve

Thank you for your interest in contributing to OpenEvolve! This document provides guidelines and instructions for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/codelion/openevolve.git`
3. Install the package in development mode: `pip install -e .`
4. Run the tests to ensure everything is working: `python -m unittest discover tests`

## Development Environment

We recommend using a virtual environment for development:

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -e ".[dev]"
```

## Pull Request Process

1. Create a new branch for your feature or bugfix: `git checkout -b feat-your-feature-name`
2. Make your changes
3. Add tests for your changes
4. Run the tests to make sure everything passes: `python -m unittest discover tests`
5. Commit your changes: `git commit -m "Add your descriptive commit message"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Submit a pull request to the main repository

## Adding Examples

We encourage adding new examples to showcase OpenEvolve's capabilities. To add a new example:

1. Create a new directory in the `examples` folder
2. Include all necessary files (initial program, evaluation code, etc.)
3. Add a README.md explaining the example
4. Make sure the example can be run with minimal setup

## Reporting Issues

When reporting issues, please include:

1. A clear description of the issue
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Environment details (OS, Python version, etc.)

## Feature Requests

Feature requests are welcome. Please provide:

1. A clear description of the feature
2. The motivation for adding this feature
3. Possible implementation ideas (if any)

## Code of Conduct

Please be respectful and considerate of others when contributing to the project. We aim to create a welcoming and inclusive environment for all contributors.
