# OpenEvolve

An open-source implementation of the AlphaEvolve system described in the Google DeepMind paper "AlphaEvolve: A coding agent for scientific and algorithmic discovery" (2025).

## Overview

OpenEvolve is an evolutionary coding agent that uses Large Language Models to optimize code through an iterative process. It orchestrates a pipeline of LLM-based code generation, evaluation, and selection to continuously improve programs for a variety of tasks.

Key features:
- Evolution of entire code files, not just single functions
- Support for multiple programming languages
- Supports OpenAI-compatible APIs for any LLM
- Multi-objective optimization
- Flexible prompt engineering
- Distributed evaluation

## How It Works

OpenEvolve follows an evolutionary approach with the following components:

1. **Prompt Sampler**: Creates context-rich prompts containing past programs, their scores, and problem descriptions
2. **LLM Ensemble**: Generates code modifications via an ensemble of language models
3. **Evaluator Pool**: Tests generated programs and assigns scores
4. **Program Database**: Stores programs and their evaluation metrics, guiding future evolution

## Getting Started

### Installation

```bash
git clone https://github.com/codelion/openevolve.git
cd openevolve
pip install -e .
```

### Quick Start

```python
from openevolve import OpenEvolve

# Initialize the system
evolve = OpenEvolve(
    initial_program="path/to/initial_program.py",
    evaluation_function="path/to/evaluator.py",
    config="path/to/config.yaml"
)

# Run the evolution
best_program = evolve.run(iterations=1000)
print(f"Best program found: {best_program.path}")
print(f"Score: {best_program.score}")
```

## Examples

See the `examples/` directory for complete examples of using OpenEvolve on various problems:
- Matrix multiplication optimization
- Packing problems
- Algorithmic discovery
- Scheduling optimization

## Citation

If you use OpenEvolve in your research, please cite:

```
@software{openevolve,
  title = {OpenEvolve: Open-source implementation of AlphaEvolve},
  author = {Asankhaya Sharma},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/codelion/openevolve}
}
```