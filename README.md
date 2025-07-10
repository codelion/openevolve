# OpenEvolve

An open-source evolutionary coding agent that began as a faithful implementation of AlphaEvolve and has evolved far beyond it, enabling automated scientific and algorithmic discovery.

![OpenEvolve Logo](openevolve-logo.png)

## Overview

OpenEvolve is an evolutionary coding agent that uses Large Language Models to automatically optimize and discover algorithms through iterative improvement. Starting from the AlphaEvolve research, it incorporates advanced features for reproducibility, multi-language support, sophisticated evaluation pipelines, and integration with cutting-edge LLM optimization techniques. It serves as both a research platform for evolutionary AI and a practical tool for automated code optimization.

### Key Features

OpenEvolve implements a comprehensive evolutionary coding system with:

- **Evolutionary Coding Agent**: LLM-guided evolution of entire code files (not just functions)
- **Distributed Controller Loop**: Asynchronous pipeline coordinating LLMs, evaluators, and databases
- **Program Database**: Storage and sampling of evolved programs with evaluation metrics
- **Prompt Sampling**: Context-rich prompts with past programs, scores, and problem descriptions
- **LLM Ensemble**: Multiple language models working together for code generation
- **Multi-objective Optimization**: Simultaneous optimization of multiple evaluation metrics
- **Checkpoint System**: Automatic saving and resuming of evolution state

#### 🔬 **Scientific Reproducibility**

- **Comprehensive Seeding**: Full deterministic reproduction with hash-based component isolation
- **Default Reproducibility**: Seed=42 by default for immediate reproducible results
- **Granular Control**: Per-component seeding for LLMs, database, and evaluation pipeline

#### 🤖 **Advanced LLM Integration**

- **Ensemble Sophistication**: Weighted model combinations with intelligent fallback strategies
- **Test-Time Compute**: Integration with [optillm](https://github.com/codelion/optillm) for Mixture of Agents (MoA) and enhanced reasoning
- **Universal API Support**: Works with any OpenAI-compatible endpoint (Anthropic, Google, local models)
- **Plugin Ecosystem**: Support for optillm plugins (readurls, executecode, z3_solver, etc.)

#### 🧬 **Evolution Algorithm Innovations**

- **MAP-Elites Implementation**: Quality-diversity algorithm for balanced exploration/exploitation
- **Island-Based Evolution**: Multiple populations with periodic migration for diversity maintenance
- **Inspiration vs Performance**: Sophisticated prompt engineering separating top performers from diverse inspirations
- **Multi-Strategy Selection**: Elite, diverse, and exploratory program sampling strategies

#### 📊 **Evaluation & Feedback Systems**

- **Artifacts Side-Channel**: Capture build errors, profiling data, and execution feedback for LLM improvement
- **Cascade Evaluation**: Multi-stage testing with progressive complexity for efficient resource usage
- **LLM-Based Feedback**: Automated code quality assessment and reasoning capture
- **Comprehensive Error Handling**: Graceful recovery from evaluation failures with detailed diagnostics

#### 🌐 **Multi-Language & Platform Support**

- **Language Agnostic**: Python, Rust, R, Metal shaders, and more
- **Platform Optimization**: Apple Silicon GPU kernels, CUDA optimization, CPU-specific tuning
- **Framework Integration**: MLX, PyTorch, scientific computing libraries

#### 🔧 **Developer Experience & Tooling**

- **Real-Time Visualization**: Interactive web-based evolution tree viewer with performance analytics
- **Advanced CLI**: Rich command-line interface with checkpoint management and configuration override
- **Comprehensive Examples**: 12+ diverse examples spanning optimization, ML, systems programming, and scientific computing
- **Error Recovery**: Robust checkpoint loading with automatic fix for common serialization issues

#### 🚀 **Performance & Scalability**

- **Threaded Parallelism**: High-throughput asynchronous evaluation pipeline
- **Resource Management**: Memory limits, timeouts, and resource monitoring
- **Efficient Storage**: Optimized database with artifact management and cleanup policies

## How It Works

OpenEvolve orchestrates a sophisticated evolutionary pipeline:

![OpenEvolve Architecture](openevolve-architecture.png)

### Core Evolution Loop

1. **Enhanced Prompt Sampler**: Creates rich prompts containing:

   - Top-performing programs (for optimization guidance)
   - Diverse inspiration programs (for creative exploration)
   - Execution artifacts and error feedback
   - Dynamic documentation fetching (via optillm plugins)

2. **Intelligent LLM Ensemble**:

   - Weighted model combinations for quality/speed tradeoffs
   - Test-time compute techniques (MoA, chain-of-thought, reflection)
   - Deterministic selection with comprehensive seeding

3. **Advanced Evaluator Pool**:

   - Multi-stage cascade evaluation
   - Artifact collection for detailed feedback
   - LLM-based code quality assessment
   - Parallel execution with resource limits

4. **Sophisticated Program Database**:
   - MAP-Elites algorithm for quality-diversity balance
   - Island-based populations with migration
   - Feature map clustering and archive management
   - Comprehensive metadata and lineage tracking

## Getting Started

### Installation

To install natively, use:

```bash
git clone https://github.com/codelion/openevolve.git
cd openevolve
pip install -e .
```

### Quick Start

#### Setting up LLM Access

OpenEvolve uses the OpenAI SDK, which means it works with any LLM provider that supports an OpenAI-compatible API:

1. **Set the API Key**: Export the `OPENAI_API_KEY` environment variable:

   ```bash
   export OPENAI_API_KEY=your-api-key-here
   ```

2. **Using Alternative LLM Providers**:
   - For providers other than OpenAI (e.g., Anthropic, Cohere, local models), update the `api_base` in your config.yaml:
   ```yaml
   llm:
     api_base: "https://your-provider-endpoint.com/v1"
   ```
3. **Maximum Flexibility with optillm**:
   - For advanced routing, rate limiting, or using multiple providers, we recommend [optillm](https://github.com/codelion/optillm)
   - optillm acts as a proxy that can route requests to different LLMs based on your rules
   - Simply point `api_base` to your optillm instance:
   ```yaml
   llm:
     api_base: "http://localhost:8000/v1"
   ```

This setup ensures OpenEvolve can work with any LLM provider - OpenAI, Anthropic, Google, Cohere, local models via Ollama/vLLM, or any OpenAI-compatible endpoint.

```python
import os
from openevolve import OpenEvolve

# Ensure API key is set
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# Initialize the system
evolve = OpenEvolve(
    initial_programs_paths=["path/to/initial_program.py"],
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

### Visualizing the evolution tree

The script in `scripts/visualize.py` allows you to visualize the evolution tree and display it in your webbrowser. The script watches live for the newest checkpoint directory in the examples/ folder structure and updates the graph. Alternatively, you can also provide a specific checkpoint folder with the `--path` parameter.

```bash
# Install requirements
pip install -r scripts/requirements.txt

# Start the visualization web server and have it watch the examples/ folder
python scripts/visualizer.py

# Start the visualization web server with a specific checkpoint
python scripts/visualizer.py --path examples/function_minimization/openevolve_output/checkpoints/checkpoint_100/
```

In the visualization UI, you can

- see the branching of your program evolution in a network visualization, with node radius chosen by the program fitness (= the currently selected metric),
- see the parent-child relationship of nodes and click through them in the sidebar (use the yellow locator icon in the sidebar to center the node in the graph),
- select the metric of interest (with the available metric choices depending on your data set),
- highlight nodes, for example the top score (for the chosen metric) or the MAP-elites members,
- click nodes to see their code and prompts (if available from the checkpoint data) in a sidebar,
- in the "Performance" tab, see their selected metric score vs generation in a graph

![OpenEvolve Visualizer](openevolve-visualizer.png)

### Docker

You can also install and execute via Docker:

```bash
docker build -t openevolve .
docker run --rm -v $(pwd):/app --network="host" openevolve examples/function_minimization/initial_program.py examples/function_minimization/evaluator.py --config examples/function_minimization/config.yaml --iterations 1000
```

## Configuration

OpenEvolve is highly configurable with advanced options:

```yaml
# Example configuration showcasing advanced features
max_iterations: 1000
random_seed: 42 # Full reproducibility by default

llm:
  # Advanced ensemble configuration
  models:
    - name: "gemini-2.0-flash-lite"
      weight: 0.7
    - name: "moa&readurls-gemini-2.0-flash" # optillm test-time compute
      weight: 0.3
  temperature: 0.7

database:
  # MAP-Elites configuration
  population_size: 500
  num_islands: 5 # Island-based evolution
  migration_interval: 20
  feature_dimensions: ["score", "complexity"] # Quality-diversity features

evaluator:
  # Advanced evaluation features
  enable_artifacts: true # Capture execution feedback
  cascade_evaluation: true # Multi-stage testing
  use_llm_feedback: true # AI-based code quality assessment

prompt:
  # Sophisticated prompt engineering
  num_top_programs: 3 # Performance examples
  num_diverse_programs: 2 # Creative inspiration
  include_artifacts: true # Execution feedback
```

Sample configuration files are available in the `configs/` directory:

- `default_config.yaml`: Comprehensive configuration with all available options
- `island_config_example.yaml`: Advanced island-based evolution setup

See the [Configuration Guide](configs/default_config.yaml) for a full list of options.

## Artifacts Channel

OpenEvolve includes an **artifacts side-channel** that allows evaluators to capture build errors, profiling results, etc. to provide better feedback to the LLM in subsequent generations. This feature enhances the evolution process by giving the LLM context about what went wrong and how to fix it.

The artifacts channel operates alongside the traditional fitness metrics.

### Example: Compilation Failure Feedback

```python
from openevolve.evaluation_result import EvaluationResult

return EvaluationResult(
    metrics={"compile_ok": 0.0, "score": 0.0},
    artifacts={
        "stderr": "SyntaxError: invalid syntax (line 15)",
        "traceback": "...",
        "failure_stage": "compilation"
    }
)
```

The next generation prompt will include:

```markdown
## Last Execution Output

### Stderr

SyntaxError: invalid syntax (line 15)

### Traceback

...
```

## Example: LLM Feedback

An example for an LLM artifact side channel is part of the default evaluation template, which ends with

```markdown
Return your evaluation as a JSON object with the following format:
{{
    "readability": [score],
    "maintainability": [score],
    "efficiency": [score],
    "reasoning": "[brief explanation of scores]"
}}
```

The non-float values, in this case the "reasoning" key of the json response that the evaluator LLM generates, will be available within the next generation prompt.

### Configuration

Artifacts can be controlled via configuration and environment variables:

```yaml
# config.yaml
evaluator:
  enable_artifacts: true

prompt:
  include_artifacts: true
  max_artifact_bytes: 4096 # 4KB limit in prompts
  artifact_security_filter: true
```

```bash
# Environment variable to disable artifacts
export ENABLE_ARTIFACTS=false
```

### Benefits

- **Faster convergence** - LLMs can see what went wrong and fix it directly
- **Better error handling** - Compilation and runtime failures become learning opportunities
- **Rich debugging context** - Full stack traces and error messages guide improvements
- **Zero overhead** - When disabled, no performance impact on evaluation

## Examples

See the `examples/` directory for complete examples of using OpenEvolve on various problems:

### Mathematical Optimization

#### [Function Minimization](examples/function_minimization/)

A comprehensive example demonstrating evolution from random search to sophisticated simulated annealing.

#### [Circle Packing](examples/circle_packing/)

Our implementation of the circle packing problem. For the n=26 case, we achieve state-of-the-art results matching published benchmarks.

<<<<<<< HEAD
Key features:

- Automatic generation of initial programs from benchmark tasks
- Evolution from simple linear models to complex mathematical expressions
- Evaluation on physics, chemistry, biology, and material science datasets
- # Competitive results compared to state-of-the-art symbolic regression methods
  Below is the optimal packing found by OpenEvolve after 800 iterations:
  > > > > > > > upstream/main

![circle-packing-result](https://github.com/user-attachments/assets/00100f9e-2ac3-445b-9266-0398b7174193)

### Advanced AI & LLM Integration

#### [Web Scraper with optillm](examples/web_scraper_optillm/)

Demonstrates integration with [optillm](https://github.com/codelion/optillm) for test-time compute optimization, including:

- **readurls plugin**: Automatic documentation fetching
- **Mixture of Agents (MoA)**: Multi-response synthesis for improved accuracy
- **Local model optimization**: Enhanced reasoning with smaller models

#### [LLM Prompt Optimization](examples/llm_prompt_optimazation/)

Evolving prompts themselves for better LLM performance, demonstrating self-improving AI systems.

### Systems & Performance Optimization

#### [MLX Metal Kernel Optimization](examples/mlx_metal_kernel_opt/)

Automated discovery of custom GPU kernels for Apple Silicon, achieving:

- **2-3x speedup** over baseline attention implementations
- **Hardware-aware optimizations** for unified memory architecture
- **Metal shader evolution** with numerical correctness validation

#### [Rust Adaptive Sort](examples/rust_adaptive_sort/)

Evolution of sorting algorithms that adapt to data patterns, showcasing OpenEvolve's language-agnostic capabilities.

### Scientific Computing & Discovery

#### [Symbolic Regression](examples/symbolic_regression/)

A comprehensive example demonstrating automated discovery of mathematical expressions from scientific datasets using the LLM-SRBench benchmark.

#### [R Robust Regression](examples/r_robust_regression/)

Developing robust regression methods resistant to outliers using R language support.

#### [Signal Processing](examples/signal_processing/)

Automated design of digital filters with superior performance characteristics.

### Web and Integration Examples

#### [Online Judge Programming](examples/online_judge_programming/)

Automated competitive programming solution generation with external evaluation systems.

#### [LM-Eval Integration](examples/lm_eval/)

Working with standard ML evaluation harnesses for automated benchmark improvement.

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
  title = {OpenEvolve: an open-source evolutionary coding agent},
  author = {Asankhaya Sharma},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/codelion/openevolve}
}
```
