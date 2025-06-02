# MLX SPDA Custom Metal Kernel Optimization - OpenEvolve Example

This example demonstrates using OpenEvolve to optimize MLX's Scaled Dot Product Attention (SPDA) using **custom Metal kernels**, similar to the kernel optimization work described in the AlphaEvolve paper. Our goal is to evolve custom Metal GPU kernels that **beat `mx.fast.scaled_dot_product_attention`** by leveraging MLX's `mx.fast.metal_kernel()` API for direct Metal C++ programming.

## Overview

### The Challenge

Modern transformer models spend most of their compute time in attention operations. Apple's MLX framework provides `mx.fast.scaled_dot_product_attention` - a highly optimized implementation that leverages Apple Silicon's unified memory and compute units. However, the AlphaEvolve paper showed that even highly optimized kernels can be improved through automated discovery.

**Our Goal**: Use OpenEvolve to discover custom Metal GPU kernels that outperform `mx.fast.scaled_dot_product_attention` by writing high-performance Metal C++ code using MLX's `mx.fast.metal_kernel()` API.

### Why This Matters

- **Real Impact**: Attention speedups directly improve transformer inference/training speed
- **Apple Silicon Optimization**: Discover patterns optimized for unified memory and ARM architecture  
- **Algorithmic Discovery**: Find novel attention patterns beyond standard implementations
- **Reproducible AlphaEvolve**: Demonstrate the paper's kernel optimization approach on an open platform

## What Gets Optimized

The evolution process optimizes custom Metal GPU kernels in the `evolved_scaled_dot_product_attention` function using MLX's `mx.fast.metal_kernel()` API:

```python
# EVOLVE-BLOCK-START
# This is what gets evolved - custom Metal C++ kernels
source = """
    template <typename T>
    [[kernel]] void fused_attention_kernel(
        const device T* q [[buffer(0)]],
        const device T* k [[buffer(1)]],
        const device T* v [[buffer(2)]],
        device T* out [[buffer(3)]],
        uint3 thread_position_in_grid [[thread_position_in_grid]]
    ) {
        // Custom optimized attention computation
        // Fuse QK^T, scaling, masking, softmax, and final matmul
        // Optimize memory access patterns for Apple Silicon
        // Use threadgroup memory and vectorization
    }
"""
kernel = mx.fast.metal_kernel(name="attention", source=source, ...)
out = kernel(inputs=[q, k, v], ...)
# EVOLVE-BLOCK-END
```

**Available Metal C++ Techniques**:
- **Kernel Fusion**: Combine QK^T + scale + mask + softmax + output in single kernel
- **Memory Optimization**: Coalesced reads, vectorized operations (float4, half4)
- **Threadgroup Memory**: Shared memory for cache optimization
- **Template Programming**: Type specialization for float16/float32
- **SIMD Operations**: Metal's built-in vectorization capabilities
- **Atomic Operations**: For complex reductions and synchronized updates
- **Tiled Computation**: Cache-friendly access patterns for large sequences

**Optimization Targets**:
- Direct Metal C++ GPU kernel programming
- Fused attention operations for reduced memory bandwidth
- Apple Silicon unified memory exploitation
- Threadgroup dispatch and synchronization optimization

**Forbidden Operations**:
- `mx.fast.*` functions (that's what we're trying to beat!)
- Only basic MLX operations without custom kernels

## Benchmark Framework

We use the provided `spda_benchmark.py` which tests across:

- **Sequence lengths**: 32 to 4096 tokens
- **Head dimensions**: 64, 80, 128  
- **Grouped Query Attention (GQA)**: Various num_kv_heads ratios
- **Mask types**: None, boolean, causal
- **Multiple configurations**: Standard and transpose layouts

The benchmark measures both **correctness** (vs reference) and **performance** (vs fused implementation).

## Expected Custom Metal Kernel Optimizations

OpenEvolve might discover:

### High-Performance Metal Kernels
- **Fused Attention Kernels**: Single kernel combining QK^T, scale, mask, softmax, and output
- **Tiled Computation**: Process attention in cache-friendly tiles using threadgroup memory
- **Vectorized Operations**: Use Metal's float4/half4 vector types for maximum throughput
- **Memory Coalescing**: Optimize memory access patterns for Apple Silicon GPU

### Apple Silicon GPU Optimizations
- **Threadgroup Strategies**: Optimal thread dispatch and synchronization patterns
- **Unified Memory Exploitation**: Leverage zero-copy between CPU and GPU
- **SIMD Utilization**: Maximum use of Apple Silicon's SIMD capabilities
- **Cache Optimization**: Metal-specific cache hierarchy utilization

### Specialized Kernel Variants
- **GQA-Optimized Kernels**: Custom kernels for grouped query attention patterns
- **Causal Mask Kernels**: Triangular computation patterns for autoregressive models
- **Sequence-Length Specialization**: Different kernels optimized for different sizes
- **Mixed Precision Kernels**: Automatic float16/float32 optimization

## Usage

### Prerequisites

```bash
# Install requirements
pip install mlx numpy pyyaml psutil

# Set up API key for LLM access (example for Gemini)
export OPENAI_API_KEY="your-api-key"  # Or appropriate API key
```

### Basic Evolution

```bash
cd examples/mlx_spda_optimization

# Run the evolution process
python ../../../openevolve-run.py initial_program.py evaluator.py --config config.yaml --iterations 150
```

### Test Initial Implementation

```bash
# Test that the initial program works
python initial_program.py

# Run evaluator on initial program
python evaluator.py
```

### Test Evolved Results

After evolution completes, test the best program against the full benchmark:

```bash
# Quick test on subset of configurations
python test_evolved.py openevolve_output/best/best_program.py --subset

# Full benchmark suite (takes longer)
python test_evolved.py openevolve_output/best/best_program.py

# Save results to file
python test_evolved.py openevolve_output/best/best_program.py --output results.txt
```

## Configuration Details

The `config.yaml` is tuned for kernel optimization:

```yaml
evolution:
  max_iterations: 150              # More iterations for complex optimization
  population_size: 80              # Large population for diverse exploration
  
llm:
  primary_model: "gemini-2.0-flash"  # Fast model for bulk generation
  secondary_model: "gemini-2.0-pro"   # Stronger model for difficult cases
  temperature: 0.9                    # Higher temp for creative optimization

evaluation:
  strategy: "cascade"                 # Quick filter + thorough evaluation
```

## Expected Results

Based on AlphaEvolve's results (23% Gemini kernel speedup), we target:

### Success Metrics
- **15-30% speedup** over `mx.fast.scaled_dot_product_attention` 
- **High accuracy** (>99% numerical agreement with reference)
- **Robustness** across different configurations (GQA, masks, sizes)
- **Consistent gains** across most benchmark configurations

### Realistic Outcomes
- **Moderate success**: 10-20% average speedup on some configurations
- **Specialized optimizations**: Large gains on specific patterns (e.g., long sequences)
- **Novel approaches**: Discovery of new attention variants
- **Negative results**: Learning what doesn't work is also valuable!

## Example Output

When successful, you'll see results like:

```
Running benchmark with evolved attention vs fused attention...
  1,   128,   128,   64,   16,   16, 0, float16,     None,  0.045,  0.052, -13.46% (speedup: 1.16x)
  1,   256,   256,   64,   16,   16, 0, float16,   causal,  0.089,  0.108, -17.59% (speedup: 1.21x)
  1,   512,   512,   64,   32,    8, 0, float16,     None,  0.178,  0.205, -13.17% (speedup: 1.15x)

Benchmark Summary:
  Average speedup: 1.18x
  Tests with speedup > 1.1x: 78%
  🎉 SUCCESS: Evolved attention achieves 1.18x average speedup!
```

## Comparison to AlphaEvolve

| Aspect | AlphaEvolve (Gemini/TPU) | This Example (MLX/Apple Silicon) |
|--------|--------------------------|-----------------------------------|
| **Target** | Pallas kernel optimization | Custom Metal kernel optimization |
| **Platform** | TPU (specialized) | Apple Silicon (unified memory) |
| **Result** | 23% speedup | Target: 15-30% speedup |
| **Impact** | 1% overall training time reduction | Direct attention speedup |
| **Constraints** | Pallas/XLA operations | Metal C++ kernel programming |
| **Method** | Evolution of tiling heuristics | Evolution of custom GPU kernels |

## Troubleshooting

### Common Issues

1. **Low accuracy scores**: 
   - Check tensor shapes and masking logic
   - Verify GQA (grouped query attention) handling
   - Test with simple configurations first

2. **Performance regressions**:
   - Start with small sequence lengths
   - Profile memory usage patterns
   - Check for unnecessary operations

3. **Evolution not converging**:
   - Increase iterations or population size
   - Adjust temperature or mutation rate
   - Check that evaluation pipeline works correctly

### Debugging

```bash
# Test specific components
python -c "from evaluator import evaluate_stage1; print(evaluate_stage1('initial_program.py'))"

# Run evaluation standalone
python evaluator.py

# Test basic functionality
python initial_program.py
```

## Advanced Usage

### Custom Test Configurations

Modify `create_test_configurations()` in `evaluator.py`:

```python
def create_test_configurations():
    return [
        # Add your custom test cases
        {"B": 1, "qsl": 2048, "ksl": 2048, "head_dim": 64, 
         "n_q_heads": 32, "n_kv_heads": 8, "dtype": "float16", "mask": "causal"},
    ]
```

### Different Tolerance Levels

Adjust accuracy requirements in `compare_attention_outputs()`:

```python
comparison = compare_attention_outputs(evolved_output, reference_output, tolerance=1e-4)
```

### Integration with Real Models

The evolved attention can potentially be integrated into MLX-based transformer implementations by replacing the attention computation while keeping the same interface.

## Scientific Value

This example demonstrates:

1. **Reproducible Research**: Open implementation of AlphaEvolve's kernel optimization approach
2. **Platform Exploration**: Understanding optimization opportunities on Apple Silicon
3. **Algorithmic Discovery**: Potential discovery of novel attention patterns
4. **Benchmarking Framework**: Systematic evaluation of attention implementations

Even negative results provide valuable insights into the limits of basic-operation optimization compared to low-level kernel optimization.

## Future Extensions

- **Mixed Precision**: Automatic precision optimization for accuracy/speed tradeoffs
- **KV Caching**: Optimize for inference patterns with key-value caching
- **Multi-Head Variants**: Explore different attention architectures
- **Cross-Platform**: Extend discoveries to other Apple Silicon variants

---

## Quick Start Summary

```bash
# 1. Install dependencies
pip install mlx numpy pyyaml psutil

# 2. Run evolution  
cd examples/mlx_spda_optimization
python ../../../openevolve-run.py initial_program.py evaluator.py --config config.yaml

# 3. Test results
python test_evolved.py openevolve_output/best/best_program.py --subset
```

This example provides a complete framework for kernel optimization research using OpenEvolve, bringing the power of AlphaEvolve's approach to the open-source community.
