# MLX Attention Optimization

This example demonstrates using OpenEvolve to optimize attention mechanisms for Apple Silicon, similar to the Gemini kernel optimization described in the AlphaEvolve paper.

## Overview

The goal is to evolve the core attention computation in MLX (Apple's ML framework) to achieve better performance while maintaining numerical accuracy. This example focuses on optimizing the scaled dot-product attention mechanism that forms the heart of transformer models.

## What Gets Optimized

The example evolves the core attention computation within the `OptimizedAttention` class:

```python
# EVOLVE-BLOCK-START
# This section contains the attention computation that gets evolved
scores = mx.matmul(queries, keys.transpose(0, 1, 3, 2))
scores = scores * self.scale
if mask is not None:
    scores = scores + mask
attn_weights = mx.softmax(scores, axis=-1)
output = mx.matmul(attn_weights, values)
# EVOLVE-BLOCK-END
```

**What remains fixed:**
- Query, Key, Value projections
- RMSNorm layers
- RoPE (Rotary Position Embedding)
- Output projection
- Input/output shapes and interfaces

**What can evolve:**
- Attention computation patterns (chunked, sparse, etc.)
- Memory access strategies
- Optimized implementations for Apple Silicon
- Alternative attention mechanisms
- Memory tiling strategies

## Key Features

### Comprehensive Evaluation
The evaluator tests multiple aspects:

1. **Numerical Accuracy**: Compares outputs with reference implementation using MLX-LM's `scaled_dot_product_attention`
2. **Performance**: Measures throughput (tokens/second) and compares with reference
3. **Memory Efficiency**: Tracks memory usage during computation
4. **Stability**: Tests with edge cases (small/large values, different input sizes)
5. **Robustness**: Tests across different configurations (batch sizes, sequence lengths, GQA)

### Test Cases
Evaluates across diverse scenarios:
- Different sequence lengths (64 to 2048 tokens)
- Various model sizes (256 to 1024 hidden dimensions)
- Grouped Query Attention (GQA) with different num_kv_heads
- Multiple batch sizes
- Edge cases for numerical stability

### Apple Silicon Optimization Opportunities
The evolution process can discover optimizations specific to Apple Silicon:
- Leveraging unified memory architecture
- Cache-friendly memory access patterns
- Vectorized operations optimized for ARM
- Efficient use of Apple's matrix units (AMX)

## Running the Example

### Prerequisites
```bash
pip install -r requirements.txt
# Or manually:
pip install mlx mlx-lm psutil numpy pyyaml
export OPENAI_API_KEY="your-api-key"  # For Gemini models
```

### Basic Usage
```bash
cd examples/mlx_attention_optimization
python ../../../openevolve-run.py initial_program.py evaluator.py --config config.yaml --iterations 200
```

### Testing Initial Implementation
```bash
python initial_program.py  # Test basic functionality
python evaluator.py       # Run full evaluation
```

## Configuration

The example uses stronger LLM models (Gemini 2.0 Flash/Pro) given the complexity of attention optimization:

```yaml
llm:
  primary_model: "gemini-2.0-flash"
  secondary_model: "gemini-2.0-pro"
  temperature: 0.8
  max_tokens: 8192
```

Key configuration choices:
- **200 iterations**: More iterations for complex optimization
- **Cascade evaluation**: Quick accuracy check before expensive performance tests
- **Larger population**: 100 programs to explore diverse optimization strategies
- **Higher temperature**: More creative exploration for novel optimizations

## Expected Optimizations

OpenEvolve might discover:

### Memory Optimizations
- **Chunked Attention**: Process attention in memory-efficient chunks
- **Tiled Computation**: Optimize memory access patterns for Apple Silicon
- **Unified Memory Exploitation**: Leverage shared CPU/GPU memory

### Algorithmic Improvements
- **Sparse Attention**: Skip computation for irrelevant token pairs
- **Local Attention**: Focus on nearby tokens for efficiency
- **Fused Operations**: Combine multiple operations to reduce memory bandwidth

### Apple Silicon Specific
- **AMX Optimization**: Efficient use of Apple's matrix units
- **Cache-Friendly Patterns**: Optimize for Apple Silicon's cache hierarchy
- **Vectorization**: Better use of NEON/Advanced SIMD instructions

## Success Metrics

A successful optimization should achieve:
- **High accuracy score** (>0.95): Maintains numerical equivalence with reference
- **Performance improvement** (>1.2x): Meaningful speedup over reference implementation  
- **Memory efficiency**: Better tokens/MB ratio
- **Stability**: Robust across different input configurations

## Comparison to AlphaEvolve Results

The original AlphaEvolve achieved:
- **23% speedup** in Gemini kernel optimization (Pallas/TPU)
- **1% overall training time reduction** for large models

Our goals for MLX/Apple Silicon:
- **15-30% attention speedup**: Similar to original results
- **Better memory efficiency**: Exploit unified memory advantages
- **Cross-model benefits**: Optimizations that work across different transformer architectures

## Using Your Optimized Attention

After evolution completes, you'll have an optimized attention implementation. Here's how to use it:

### Quick Start (3 lines of code)
```python
from attention_integration import load_and_patch_model
from mlx_lm import generate

# Load any MLX-LM model with evolved attention  
model, tokenizer = load_and_patch_model(
    model_path="mlx-community/Qwen3-0.6B-bf16",
    evolved_program_path="openevolve_output/best/best_program.py"
)

# Use exactly like any other MLX-LM model - but faster!
response = generate(model, tokenizer, "Write a Python function:", max_tokens=100)
```

### Testing Your Implementation
```bash
# Quick demo
python use_evolved_attention.py demo

# Comprehensive benchmarking
python test_workloads.py --model mlx-community/Qwen3-0.6B-bf16 --evolved-program openevolve_output/best/best_program.py
```

### Recommended Test Workloads
- **Text generation**: Stories, articles, reports (15-30% speedup expected)
- **Code generation**: Functions, classes, APIs (20-40% speedup expected)
- **Long-form content**: 1024+ tokens (30-50% speedup expected)
- **Question answering**: Complex reasoning tasks (10-25% speedup expected)

📖 **See [USAGE.md](USAGE.md) for complete integration guide and benchmarking instructions.**

## Advanced Usage

### Custom Test Cases
Modify `create_test_cases()` in `evaluator.py` to test specific configurations:

```python
def create_test_cases():
    return [
        {"batch_size": 1, "seq_len": 4096, "hidden_size": 2048, "num_heads": 32, "num_kv_heads": 8},
        # Add your custom test cases
    ]
```

### Different Tolerance Levels
Adjust accuracy requirements in `compare_outputs()`:

```python
comparison = compare_outputs(evolved_output, reference_output, tolerance=1e-4)
```

### Integration Testing
Test evolved attention with real models by replacing the attention module in mlx-lm implementations.

## Troubleshooting

### Common Issues
1. **Low accuracy scores**: Check tensor shapes and ensure proper masking
2. **Memory errors**: Reduce batch sizes or sequence lengths in test cases
3. **Slow evaluation**: Reduce number of test cases or performance benchmark runs

### Debugging
Enable detailed logging:
```bash
python evaluator.py  # Run standalone evaluation
```

Check specific test cases:
```python
python -c "
from evaluator import evaluate_stage1
print(evaluate_stage1('initial_program.py'))
"
```

## Future Extensions

- **Multi-Head Attention Variants**: Optimize different attention patterns
- **KV Caching**: Optimize for inference with key-value caching
- **Mixed Precision**: Automatic precision optimization
- **Cross-Platform**: Extend optimizations to other Apple Silicon variants (A-series, etc.)
