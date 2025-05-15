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
    initial_program_path="path/to/initial_program.py",
    evaluation_file="path/to/evaluator.py",
    config_path="path/to/config.yaml"
)

# Run the evolution
best_program = await evolve.run(iterations=1000)
print(f"Best program metrics:")
for name, value in best_program.metrics.items():
    print(f"  {name}: {value:.4f}")
```

### Command-Line Usage

OpenEvolve can also be run from the command line:

```bash
python openevolve-run.py path/to/initial_program.py path/to/evaluator.py --config path/to/config.yaml --iterations 1000
```

## Configuration

OpenEvolve is highly configurable. You can specify configuration options in a YAML file:

```yaml
# Example configuration
max_iterations: 1000
llm:
  primary_model: "gemini-2.0-flash-lite"
  secondary_model: "gemini-2.0-flash"
  temperature: 0.7
database:
  population_size: 500
  num_islands: 5
```

Sample configuration files are available in the `configs/` directory:
- `default_config.yaml`: Comprehensive configuration with all available options
- `matrix_multiplication_config.yaml`: Configuration optimized for matrix multiplication
- `min_max_distance_config.yaml`: Configuration for geometric optimization

See the [Configuration Guide](configs/default_config.yaml) for a full list of options.

## Examples

See the `examples/` directory for complete examples of using OpenEvolve on various problems:

### Matrix Multiplication Optimization
Evolves more efficient matrix multiplication algorithms:
```bash
cd examples/matrix_multiplication
python optimize.py --iterations 100
```

### Min-Max Distance Optimization
Finds optimal point configurations that minimize the ratio of maximum to minimum distances:
```bash
cd examples/min_max_distance
python optimize.py --iterations 150 --num-points 16
```

## Preparing Your Own Problems

To use OpenEvolve for your own problems:

1. **Mark code sections** to evolve with `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` comments
2. **Create an evaluation function** that returns a dictionary of metrics
3. **Configure OpenEvolve** with appropriate parameters
4. **Run the evolution** process

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