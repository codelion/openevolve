# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OpenEvolve is an open-source implementation of Google DeepMind's AlphaEvolve system - an evolutionary coding agent that uses LLMs to optimize code through iterative evolution. The framework can evolve code in multiple languages (Python, R, Rust, etc.) for tasks like scientific computing, optimization, and algorithm discovery.

## Essential Commands

### Development Setup
```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"

# Or use Makefile
make install
```

### Running Tests
```bash
# Run unit tests only (fast, no LLM required)
python -m unittest discover tests

# Or use Makefile
make test

# Run integration tests (requires optillm)
make test-integration

# Run all tests
make test-all

# Run single test file
python -m unittest tests.test_database

# Run single test case
python -m unittest tests.test_database.TestProgramDatabase.test_add_and_get
```

**Note**: Unit tests require `OPENAI_API_KEY` environment variable to be set (can be any placeholder value like `test-key`). Integration tests need optillm server running.

### Code Formatting
```bash
# Format with Black (line length: 100)
python -m black openevolve examples tests scripts

# Or use Makefile
make lint
```

### Running OpenEvolve
```bash
# Basic evolution run
python openevolve-run.py path/to/initial_program.py path/to/evaluator.py --config path/to/config.yaml --iterations 1000

# Resume from checkpoint
python openevolve-run.py path/to/initial_program.py path/to/evaluator.py \
  --config path/to/config.yaml \
  --checkpoint path/to/checkpoint_directory \
  --iterations 50

# Using the CLI entry point (installed via pip)
openevolve-run path/to/initial_program.py path/to/evaluator.py --config path/to/config.yaml --iterations 1000
```

### Visualization
```bash
# Install visualization dependencies first
pip install -r scripts/requirements.txt

# View evolution tree
python scripts/visualizer.py --path examples/function_minimization/openevolve_output/checkpoints/checkpoint_100/
```

## High-Level Architecture

### Core Components

1. **Controller (`openevolve/controller.py`)**: Main orchestrator that manages the evolution process:
   - Coordinates evolution loop and checkpointing
   - Manages ProcessPoolExecutor for parallel iteration execution
   - Handles graceful shutdown and state persistence

2. **Process Parallel (`openevolve/process_parallel.py`)**: True parallel execution layer:
   - Worker pool with process-based isolation
   - Each worker loads database snapshot for independent evolution
   - Lazy initialization of LLM/evaluator components per worker
   - Preserves parent environment variables in child processes

3. **Database (`openevolve/database.py`)**: Implements MAP-Elites algorithm with island-based evolution:
   - Programs mapped to multi-dimensional feature grid (`Program` dataclass)
   - Multiple isolated populations (islands) evolve independently
   - Periodic migration between islands prevents convergence (lazy migration based on generation counts)
   - Tracks absolute best program separately (`best_program_id`)
   - Per-island best tracking (`island_best_programs`)
   - Feature binning can be uniform (int) or per-dimension (dict)

4. **Evaluator (`openevolve/evaluator.py`)**: Cascade evaluation pattern:
   - Stage 1: Quick validation (syntax/imports)
   - Stage 2: Basic performance testing
   - Stage 3: Comprehensive evaluation
   - Programs must pass thresholds at each stage
   - Supports timeout protection and artifact collection

5. **LLM Integration (`openevolve/llm/`)**: Ensemble approach with multiple models:
   - Weighted model selection from configured models
   - Async generation with retry logic and fallback
   - Configurable API base for any OpenAI-compatible endpoint
   - Separate evaluator models for LLM-based code quality assessment

6. **Iteration (`openevolve/iteration.py`)**: Worker process that:
   - Samples programs from islands using various strategies
   - Generates mutations via LLM with prompt context
   - Evaluates programs through cascade stages
   - Stores artifacts (JSON or files based on size threshold)

### Key Architectural Patterns

- **Island-Based Evolution**: Multiple populations evolve separately with periodic migration
- **MAP-Elites**: Maintains diversity by mapping programs to feature grid cells
- **Artifact System**: Side-channel for programs to return debugging data, stored as JSON or files
- **Process Worker Pattern**: Each iteration runs in fresh process with database snapshot
- **Double-Selection**: Programs for inspiration differ from those shown to LLM
- **Lazy Migration**: Islands migrate based on generation counts, not iterations

### Code Evolution Markers

Mark code sections to evolve using:
```python
# EVOLVE-BLOCK-START
# Code to evolve goes here
# EVOLVE-BLOCK-END
```

### Configuration

YAML-based configuration with hierarchical structure:
- LLM models and parameters
- Evolution strategies (diff-based vs full rewrites)
- Database and island settings
- Evaluation parameters

### Important Patterns

1. **Checkpoint/Resume**: Automatic saving of entire system state with seamless resume capability
2. **Parallel Evaluation**: Multiple programs evaluated concurrently via TaskPool
3. **Error Resilience**: Individual failures don't crash system - extensive retry logic and timeout protection
4. **Prompt Engineering**: Template-based system with context-aware building and evolution history

### Library API

OpenEvolve can be used as a Python library (see `openevolve/api.py`):

```python
from openevolve import run_evolution, evolve_function, EvolutionResult

# Using file paths
result = run_evolution(
    initial_program='program.py',
    evaluator='evaluator.py',
    config='config.yaml',
    iterations=100
)

# Using inline code
result = run_evolution(
    initial_program='''
        # EVOLVE-BLOCK-START
        def solve(x): return x * 2
        # EVOLVE-BLOCK-END
    ''',
    evaluator=lambda path: {"score": benchmark(path)},
    iterations=100
)

# Evolve Python functions directly
def bubble_sort(arr): ...
result = evolve_function(
    bubble_sort,
    test_cases=[([3,1,2], [1,2,3])],
    iterations=50
)
```

### Development Notes

- Python >=3.10 required (uses dataclasses, type hints)
- Uses OpenAI-compatible APIs for LLM integration (configurable via `api_base`)
- Tests use unittest framework (pytest for integration tests)
- Black for code formatting (line length: 100)
- Artifacts threshold: Small (<10KB) stored in DB as JSON, large saved to disk
- Process workers load database snapshots for true parallelism (no shared state)
- Config uses YAML with hierarchical dataclass structure (`Config`, `DatabaseConfig`, `LLMConfig`, etc.)
- All components are seeded for reproducibility (default seed=42)

## Working with Examples

Each example in `examples/` follows a standard structure:

```
examples/example_name/
├── README.md              # Explains the example
├── config.yaml            # Evolution configuration
├── initial_program.py     # Starting program to evolve
├── evaluator.py           # Evaluation logic
└── requirements.txt       # (optional) Additional dependencies
```

### Creating a New Example

1. Copy an existing example as template (e.g., `function_minimization/`)
2. Modify `initial_program.py` with code to evolve (mark with `# EVOLVE-BLOCK-START/END`)
3. Write evaluator that returns metrics dict: `{"score": float, ...}`
4. Configure `config.yaml` with appropriate system message and parameters
5. Test with small iteration count first

### Evaluator Requirements

Evaluators must accept a program path and return metrics:

```python
def evaluate(program_path: str) -> Union[Dict[str, float], EvaluationResult]:
    # Option 1: Return dict with metrics
    return {"score": 0.85, "accuracy": 0.9}

    # Option 2: Return EvaluationResult with artifacts
    from openevolve.evaluation_result import EvaluationResult
    return EvaluationResult(
        metrics={"score": 0.85},
        artifacts={"stderr": "...", "llm_feedback": "..."}
    )
```

## Configuration Deep Dive

### Key Config Parameters

- `max_iterations`: Total evolution iterations
- `random_seed`: For reproducibility (set to same value for identical runs)
- `output_dir`: Where to save checkpoints and results

### LLM Configuration

```yaml
llm:
  api_base: "https://api.openai.com/v1"  # Or Gemini, Ollama, etc.
  models:
    - name: "gpt-4"
      weight: 0.6
    - name: "gpt-3.5-turbo"
      weight: 0.4
  temperature: 0.7
  max_tokens: 4096
```

### Island Evolution Settings

```yaml
database:
  num_islands: 5              # More = more diversity, slower convergence
  migration_interval: 50      # Generations between migrations (not iterations!)
  migration_rate: 0.1         # Fraction of top programs to migrate
  feature_dimensions:         # Quality-diversity dimensions
    - "complexity"            # Built-in: code length
    - "diversity"             # Built-in: edit distance
    - "custom_metric"         # Custom: returned from evaluator
```

**Important**: `migration_interval` is in generations, not iterations. Each island tracks its own generation count independently.

### Prompt Configuration

```yaml
prompt:
  system_message: |
    You are an expert programmer...
  num_top_programs: 3         # Best performers for inspiration
  num_diverse_programs: 2     # Diverse programs for exploration
  use_template_stochasticity: true  # Randomize prompt templates
```

## Troubleshooting

### Common Issues

1. **OPENAI_API_KEY not set**: Even for unit tests, set to placeholder value
2. **Evolution gets stuck**: Increase `num_diverse_programs` or add more islands
3. **Worker errors**: Check that evaluator doesn't use unpicklable objects (lambdas, local classes)
4. **Memory issues**: Reduce `num_parallel_workers` or `archive_size`
5. **Slow evolution**: Enable `cascade_evaluation` to filter bad programs early

### Debugging Tips

- Enable artifacts to see program errors: `evaluator.enable_artifacts: true`
- Check checkpoint files in `output_dir/checkpoints/` for saved state
- Use `--checkpoint` flag to resume from last successful checkpoint
- Lower `num_parallel_workers` to 1 for easier debugging
- Check `openevolve_output/evolution.log` for detailed execution logs