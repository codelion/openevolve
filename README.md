# OpenEvolve

An open-source implementation of the AlphaEvolve system described in the Google DeepMind paper "AlphaEvolve: A coding agent for scientific and algorithmic discovery" (2025).

![OpenEvolve Logo](openevolve-logo.png)

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

![OpenEvolve Architecture](openevolve-architecture.png)

1. **Prompt Sampler**: Creates context-rich prompts containing past programs, their scores, and problem descriptions
2. **LLM Ensemble**: Generates code modifications via an ensemble of language models
3. **Evaluator Pool**: Tests generated programs and assigns scores
4. **Program Database**: Stores programs and their evaluation metrics, guiding future evolution

The controller orchestrates interactions between these components in an asynchronous pipeline, maximizing throughput to evaluate as many candidate solutions as possible.

## Getting Started

### Installation

To install natively, use:
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

### Resuming from Checkpoints

OpenEvolve automatically saves checkpoints at intervals specified by the `checkpoint_interval` config parameter (default is 10 iterations). You can resume an evolution run from a saved checkpoint:

```bash
python openevolve-run.py path/to/initial_program.py path/to/evaluator.py \
  --config path/to/config.yaml \
  --checkpoint path/to/checkpoint_directory \
  --iterations 50
```

When resuming from a checkpoint:
- The system loads all previously evolved programs and their metrics
- Checkpoint numbering continues from where it left off (e.g., if loaded from checkpoint_50, the next checkpoint will be checkpoint_60)
- All evolution state is preserved (best programs, feature maps, archives, etc.)
- Each checkpoint directory contains a copy of the best program at that point in time

Example workflow with checkpoints:

```bash
# Run for 50 iterations (creates checkpoints at iterations 10, 20, 30, 40, 50)
python openevolve-run.py examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --iterations 50

# Resume from checkpoint 50 for another 50 iterations (creates checkpoints at 60, 70, 80, 90, 100)
python openevolve-run.py examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --checkpoint examples/function_minimization/openevolve_output/checkpoints/checkpoint_50 \
  --iterations 50
```

### Comparing Results Across Checkpoints

Each checkpoint directory contains the best program found up to that point, making it easy to compare solutions over time:

```
checkpoints/
  checkpoint_10/
    best_program.py         # Best program at iteration 10
    best_program_info.json  # Metrics and details
    programs/               # All programs evaluated so far
    metadata.json           # Database state
  checkpoint_20/
    best_program.py         # Best program at iteration 20
    ...
```

You can compare the evolution of solutions by examining the best programs at different checkpoints:

```bash
# Compare best programs at different checkpoints
diff -u checkpoints/checkpoint_10/best_program.py checkpoints/checkpoint_20/best_program.py

# Compare metrics
cat checkpoints/checkpoint_*/best_program_info.json | grep -A 10 metrics
```
### Docker

You can also install and execute via Docker:
```bash
docker build -t openevolve .
docker run --rm -v .:/app openevolve examples/function_minimization/initial_program.py examples/function_minimization/evaluator.py --config examples/function_minimization/config.yaml --iterations 1000
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

See the [Configuration Guide](configs/default_config.yaml) for a full list of options.

## Examples

See the `examples/` directory for complete examples of using OpenEvolve on various problems:

### Circle Packing

Our implementation of the circle packing problem from the AlphaEvolve paper, where we successfully match their reported results within 0.04%.

[Explore the Circle Packing Example](examples/circle_packing/)

We have sucessfully replicated the results from the AlphaEvolve paper, below is the packing found by OpenEvolve after 800 iterations

![alpha-evolve-replication](https://github.com/user-attachments/assets/00100f9e-2ac3-445b-9266-0398b7174193)

This is exactly the packing reported by AlphaEolve in their paper (Figure 14): 

![alpha-evolve-results](https://github.com/user-attachments/assets/0c9affa5-053d-404e-bb2d-11479ab248c9)

### Function Minimization

An example showing how OpenEvolve can transform a simple random search algorithm into a sophisticated simulated annealing approach.

[Explore the Function Minimization Example](examples/function_minimization/)

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
