# OpenEvolve Examples

This directory contains a collection of examples demonstrating how to use OpenEvolve for various tasks including optimization, algorithm discovery, and code evolution. Each example showcases different aspects of OpenEvolve's capabilities and provides templates for creating your own evolutionary coding projects.

## Quick Start Template

To create your own OpenEvolve example, you need three essential components:

### 1. Initial Program (`initial_program.py`)

Your initial program must contain exactly **one** `EVOLVE-BLOCK`:

```python
# EVOLVE-BLOCK-START
def your_function():
    # Your initial implementation here
    # This is the only section OpenEvolve will modify
    pass
# EVOLVE-BLOCK-END

# Helper functions and other code outside the evolve block
def helper_function():
    # This code won't be modified by OpenEvolve
    pass
```

**Critical Requirements:**
- ✅ **Exactly one EVOLVE-BLOCK** (not multiple blocks)
- ✅ Use `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` markers
- ✅ Put only the code you want evolved inside the block
- ✅ Helper functions and imports go outside the block

### 2. Evaluator (`evaluator.py`)

Your evaluator must return a **dictionary** with specific metric names:

```python
def evaluate(program_path: str) -> Dict:
    """
    Evaluate the program and return metrics as a dictionary.
    
    CRITICAL: Must return a dictionary, not an EvaluationResult object.
    """
    try:
        # Import and run your program
        # Calculate metrics
        
        return {
            'combined_score': 0.8,  # PRIMARY METRIC for evolution (required)
            'accuracy': 0.9,        # Your custom metrics
            'speed': 0.7,
            'robustness': 0.6,
            # Add any other metrics you want to track
        }
    except Exception as e:
        return {
            'combined_score': 0.0,  # Always return combined_score, even on error
            'error': str(e)
        }
```

**Critical Requirements:**
- ✅ **Return a dictionary**, not `EvaluationResult` object
- ✅ **Must include `'combined_score'`** - this is the primary metric OpenEvolve uses
- ✅ Higher `combined_score` values should indicate better programs
- ✅ Handle exceptions and return `combined_score: 0.0` on failure

### 3. Configuration (`config.yaml`)

Essential configuration structure:

```yaml
# Evolution settings
max_iterations: 100
checkpoint_interval: 10
parallel_evaluations: 1

# LLM configuration
llm:
  api_base: "https://api.openai.com/v1"  # Or your LLM provider
  models:
    - name: "gpt-4"
      weight: 1.0
  temperature: 0.7
  max_tokens: 4000
  timeout: 120

# Database configuration (MAP-Elites algorithm)
database:
  population_size: 50
  num_islands: 3
  migration_interval: 10
  feature_dimensions:  # MUST be a list, not an integer
    - "score"
    - "complexity"

# Evaluation settings
evaluator:
  timeout: 60
  max_retries: 3

# Prompt configuration
prompt:
  system_message: |
    You are an expert programmer. Your goal is to improve the code
    in the EVOLVE-BLOCK to achieve better performance on the task.
    
    Focus on algorithmic improvements and code optimization.
  num_top_programs: 3
  num_diverse_programs: 2

# Logging
log_level: "INFO"
```

**Critical Requirements:**
- ✅ **`feature_dimensions` must be a list** (e.g., `["score", "complexity"]`), not an integer
- ✅ Set appropriate timeouts for your use case
- ✅ Configure LLM settings for your provider
- ✅ Use meaningful `system_message` to guide evolution

## Common Configuration Mistakes

❌ **Wrong:** `feature_dimensions: 2`  
✅ **Correct:** `feature_dimensions: ["score", "complexity"]`

❌ **Wrong:** Returning `EvaluationResult` object  
✅ **Correct:** Returning `{'combined_score': 0.8, ...}` dictionary

❌ **Wrong:** Using `'total_score'` metric name  
✅ **Correct:** Using `'combined_score'` metric name

❌ **Wrong:** Multiple EVOLVE-BLOCK sections  
✅ **Correct:** Exactly one EVOLVE-BLOCK section

## Running Your Example

```bash
# Basic run
python openevolve-run.py path/to/initial_program.py path/to/evaluator.py --config path/to/config.yaml --iterations 100

# Resume from checkpoint
python openevolve-run.py path/to/initial_program.py path/to/evaluator.py \
  --config path/to/config.yaml \
  --checkpoint path/to/checkpoint_directory \
  --iterations 50

# View results
python scripts/visualizer.py --path path/to/openevolve_output/checkpoints/checkpoint_100/
```

## Advanced Configuration Options

### LLM Ensemble (Multiple Models)
```yaml
llm:
  models:
    - name: "gpt-4"
      weight: 0.7
    - name: "claude-3-sonnet"
      weight: 0.3
```

### Island Evolution (Population Diversity)
```yaml
database:
  num_islands: 5        # More islands = more diversity
  migration_interval: 15  # How often islands exchange programs
  population_size: 100   # Larger population = more exploration
```

### Cascade Evaluation (Multi-Stage Testing)
```yaml
evaluator:
  cascade_stages:
    - stage1_timeout: 30   # Quick validation
    - stage2_timeout: 120  # Full evaluation
```

## Example Directory

### 🧮 Mathematical Optimization

#### [Function Minimization](function_minimization/)
**Task:** Find global minimum of complex non-convex function  
**Achievement:** Evolved from random search to sophisticated simulated annealing  
**Key Lesson:** Shows automatic discovery of optimization algorithms  
```bash
cd examples/function_minimization
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

#### [Circle Packing](circle_packing/)
**Task:** Pack 26 circles in unit square to maximize sum of radii  
**Achievement:** Matched AlphaEvolve paper results (2.634/2.635)  
**Key Lesson:** Demonstrates evolution from geometric heuristics to mathematical optimization  
```bash
cd examples/circle_packing
python ../../openevolve-run.py initial_program.py evaluator.py --config config_phase_1.yaml
```

### 🔧 Algorithm Discovery

#### [Signal Processing](signal_processing/)
**Task:** Design digital filters for audio processing  
**Achievement:** Discovered novel filter designs with superior characteristics  
**Key Lesson:** Shows evolution of domain-specific algorithms  
```bash
cd examples/signal_processing
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

#### [Rust Adaptive Sort](rust_adaptive_sort/)
**Task:** Create sorting algorithm that adapts to data patterns  
**Achievement:** Evolved sorting strategies beyond traditional algorithms  
**Key Lesson:** Multi-language support (Rust) and algorithm adaptation  
```bash
cd examples/rust_adaptive_sort
python ../../openevolve-run.py initial_program.rs evaluator.py --config config.yaml
```

### 🚀 Performance Optimization

#### [MLX Metal Kernel Optimization](mlx_metal_kernel_opt/)
**Task:** Optimize attention mechanisms for Apple Silicon  
**Achievement:** 2-3x speedup over baseline implementation  
**Key Lesson:** Hardware-specific optimization and performance tuning  
```bash
cd examples/mlx_metal_kernel_opt
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

### 🌐 Web and Data Processing

#### [Web Scraper with optillm](web_scraper_optillm/)
**Task:** Extract API documentation from HTML pages  
**Achievement:** Demonstrates optillm integration with readurls and MoA  
**Key Lesson:** Shows integration with LLM proxy systems and test-time compute  
```bash
cd examples/web_scraper_optillm
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

### 💻 Programming Challenges

#### [Online Judge Programming](online_judge_programming/)
**Task:** Solve competitive programming problems  
**Achievement:** Automated solution generation and submission  
**Key Lesson:** Integration with external evaluation systems  
```bash
cd examples/online_judge_programming
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

### 📊 Machine Learning and AI

#### [LLM Prompt Optimization](llm_prompt_optimazation/)
**Task:** Evolve prompts for better LLM performance  
**Achievement:** Discovered effective prompt engineering techniques  
**Key Lesson:** Self-improving AI systems and prompt evolution  
```bash
cd examples/llm_prompt_optimazation
python ../../openevolve-run.py initial_prompt.txt evaluator.py --config config.yaml
```

#### [LM-Eval Integration](lm_eval/)
**Task:** Integrate with language model evaluation harness  
**Achievement:** Automated benchmark improvement  
**Key Lesson:** Integration with standard ML evaluation frameworks  

#### [Symbolic Regression](symbolic_regression/)
**Task:** Discover mathematical expressions from data  
**Achievement:** Automated discovery of scientific equations  
**Key Lesson:** Scientific discovery and mathematical modeling  

### 🔬 Scientific Computing

#### [R Robust Regression](r_robust_regression/)
**Task:** Develop robust statistical regression methods  
**Achievement:** Novel statistical algorithms resistant to outliers  
**Key Lesson:** Multi-language support (R) and statistical algorithm evolution  
```bash
cd examples/r_robust_regression
python ../../openevolve-run.py initial_program.r evaluator.py --config config.yaml
```

### 🎯 Advanced Features

#### [Circle Packing with Artifacts](circle_packing_with_artifacts/)
**Task:** Circle packing with detailed execution feedback  
**Achievement:** Advanced debugging and artifact collection  
**Key Lesson:** Using OpenEvolve's artifact system for detailed analysis  
```bash
cd examples/circle_packing_with_artifacts
python ../../openevolve-run.py initial_program.py evaluator.py --config config_phase_1.yaml
```

## Best Practices

### 🎯 Design Effective Evaluators
- Use meaningful metrics that reflect your goals
- Include both quality and efficiency measures
- Handle edge cases and errors gracefully
- Provide informative feedback for debugging

### 🔧 Configuration Tuning
- Start with smaller populations and fewer iterations for testing
- Increase `num_islands` for more diverse exploration
- Adjust `temperature` based on how creative you want the LLM to be
- Set appropriate timeouts for your compute environment

### 📈 Evolution Strategy
- Use multiple phases with different configurations
- Begin with exploration, then focus on exploitation
- Consider cascade evaluation for expensive tests
- Monitor progress and adjust configuration as needed

### 🐛 Debugging
- Check logs in `openevolve_output/logs/`
- Examine failed programs in checkpoint directories
- Use artifacts to understand program behavior
- Test your evaluator independently before evolution

## Getting Help

- 📖 See individual example READMEs for detailed walkthroughs
- 🔍 Check the main [OpenEvolve documentation](../README.md)
- 💬 Open issues on the [GitHub repository](https://github.com/codelion/openevolve)

Each example is self-contained and includes all necessary files to get started. Pick an example similar to your use case and adapt it to your specific problem!