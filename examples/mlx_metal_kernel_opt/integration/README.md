# MLX Metal Kernel Optimization Integration

This package provides seamless integration of optimized Metal kernels with MLX-LM, delivering significant performance improvements for transformer attention computations on Apple Silicon.

## 🚀 Key Features

- **Intelligent Dispatch**: Automatically detects model architecture and applies appropriate optimizations
- **Graceful Fallback**: Falls back to standard MLX operations when optimizations aren't beneficial
- **Multiple Attention Patterns**: Supports GQA, MQA, and MHA with pattern-specific optimizations
- **Easy Integration**: Simple monkey-patching for existing mlx-lm code
- **Comprehensive Benchmarking**: Built-in tools for performance measurement and comparison
- **Apple Silicon Optimized**: Leverages Metal Performance Shaders and unified memory architecture

## 📊 Performance Improvements

| Model Type | Architecture | Expected Speedup | Memory Reduction |
|------------|--------------|------------------|------------------|
| Qwen3      | 40:8 GQA     | 1.5-2.0x        | 10-15%          |
| Llama-3    | 32:8 GQA     | 1.3-1.8x        | 8-12%           |
| Gemma      | 24:24 MHA    | 1.2-1.5x        | 5-10%           |
| Mistral    | 32:8 GQA     | 1.4-1.9x        | 8-12%           |

## 🛠 Installation

1. **Prerequisites**:
   ```bash
   pip install mlx mlx-lm
   ```

2. **Integration Setup**:
   ```bash
   # Copy the integration folder to your project
   cp -r integration/ /path/to/your/project/
   ```

## 🔧 Quick Start

### Basic Usage

```python
from integration import patch_mlx_lm, unpatch_mlx_lm
from mlx_lm import load, generate

# Apply optimizations
patch_mlx_lm(enable_debug=True)

# Use mlx-lm normally - optimizations applied automatically
model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
response = generate(model, tokenizer, prompt="Hello!", max_tokens=100)

# Remove optimizations when done
unpatch_mlx_lm()
```

### Context Manager Pattern

```python
from integration.mlx_lm_integration import MLXLMIntegration

class OptimizedMLX:
    def __enter__(self):
        self.patched_count = patch_mlx_lm(enable_debug=False)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        unpatch_mlx_lm(enable_debug=False)

# Optimizations applied only within this block
with OptimizedMLX():
    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    response = generate(model, tokenizer, prompt="Hello!", max_tokens=100)
# Optimizations automatically removed
```

### Custom Configuration

```python
from integration import configure_optimizer, patch_mlx_lm

# Configure optimization thresholds
configure_optimizer(
    enable_debug=True,
    min_seq_len=128,        # Lower threshold for short sequences
    max_seq_len=4096,       # Higher limit for long sequences
    gqa_ratio_min=3,        # Require at least 3:1 GQA ratio
    min_heads=16            # Require at least 16 heads
)

# Apply with custom configuration
patch_mlx_lm()
```

## 🧪 Testing and Benchmarking

### Quick Demo

```bash
python integration/demo_integration.py --quick-test
```

### Interactive Demo

```bash
python integration/demo_integration.py --interactive --model qwen2.5-0.5b
```

### Comprehensive Benchmark

```bash
python integration/demo_integration.py --comprehensive
```

### Usage Examples

```bash
python integration/usage_examples.py
```

## 📈 Monitoring Performance

### Check Optimization Status

```python
from integration import get_integration_status

status = get_integration_status()
print(f"Patched: {status['is_patched']}")
print(f"Optimization rate: {status['optimizer_stats']['optimization_rate']:.1%}")
```

### Benchmark Specific Models

```python
from integration import benchmark_optimization

results = benchmark_optimization(
    model_name="qwen3",
    seq_lengths=[256, 512, 1024, 2048],
    warmup_runs=3,
    benchmark_runs=10,
    save_results=True
)

for result in results:
    print(f"Seq {result.seq_length}: {result.speedup:.2f}x speedup")
```

## 🎯 Supported Models

| Model Family | Pattern | Priority | Status |
|--------------|---------|----------|--------|
| Qwen3        | GQA 5:1 | High     | ✅ Optimized |
| Qwen2        | GQA 4:1 | High     | ✅ Optimized |
| Llama-3      | GQA 4:1 | High     | ✅ Optimized |
| Mistral      | GQA 4:1 | High     | ✅ Optimized |
| Gemma        | MHA 1:1 | Medium   | ✅ Optimized |
| Phi-3        | GQA 4:1 | Medium   | ✅ Optimized |
| DeepSeek-V3  | GQA     | High     | ✅ Optimized |

## ⚙️ How It Works

### 1. Attention Pattern Detection

The optimizer automatically detects attention patterns:

```python
config = AttentionConfig(
    num_heads=40,
    num_kv_heads=8,
    head_dim=128,
    seq_len=1024,
    batch_size=1
)

# Automatically detects: GQA-5:1 pattern
print(config.attention_pattern)  # "GQA-5:1"
```

### 2. Intelligent Dispatch

Based on the detected pattern and thresholds:

```python
should_optimize, reason = optimizer.should_optimize(config)
if should_optimize:
    # Apply optimized Metal kernel
    result = optimized_attention(queries, keys, values, scale, mask)
else:
    # Fall back to standard MLX implementation
    result = mx.fast.scaled_dot_product_attention(queries, keys, values, scale, mask)
```

### 3. Metal Kernel Optimization

The Metal kernels include:

- **Memory Coalescing**: Optimized memory access patterns for Apple Silicon
- **SIMD Vectorization**: 4-way and 8-way vectorized operations
- **Online Softmax**: Memory-efficient attention computation
- **Pattern-Specific Logic**: GQA head mapping, MQA single-head optimization

## 🔍 Technical Details

### Optimization Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_seq_len` | 64 | Minimum sequence length for optimization |
| `max_seq_len` | 4096 | Maximum supported sequence length |
| `min_head_dim` | 64 | Minimum head dimension for vectorization |
| `max_head_dim` | 256 | Maximum supported head dimension |
| `min_heads` | 8 | Minimum number of heads for optimization |
| `gqa_ratio_min` | 2 | Minimum GQA ratio to trigger optimization |

### Metal Kernel Features

1. **GQA Optimization**:
   - Efficient head mapping for grouped queries
   - Optimized memory layout for KV head sharing
   - Vectorized computation with loop unrolling

2. **MQA Optimization**:
   - Single KV head specialized kernel
   - Reduced memory bandwidth requirements
   - Optimized for single-head broadcasting

3. **MHA Optimization**:
   - Standard multi-head attention with vectorization
   - Memory-efficient implementation
   - SIMD optimizations for large head counts

## 🐛 Troubleshooting

### Common Issues

1. **No Optimization Applied**:
   ```python
   # Check if model meets thresholds
   status = get_integration_status()
   print(status['optimizer_stats'])
   ```

2. **Fallback to Standard Implementation**:
   ```python
   # Enable debug to see fallback reasons
   patch_mlx_lm(enable_debug=True)
   ```

3. **Memory Issues**:
   ```python
   # Lower sequence length threshold
   configure_optimizer(max_seq_len=2048)
   ```

### Debug Mode

Enable debug output to see optimization decisions:

```python
patch_mlx_lm(enable_debug=True)
# Output will show:
# ✅ Patched qwen3 attention
# ⚡ Applying GQA-5:1 optimization: GQA pattern with 5:1 ratio benefits from custom kernel
# 🔄 Falling back to MLX SDPA: Sequence length 32 below threshold 64
```

## 📋 API Reference

### Main Functions

- `patch_mlx_lm(enable_debug=False, **kwargs)` - Apply optimizations
- `unpatch_mlx_lm(enable_debug=False)` - Remove optimizations  
- `get_integration_status()` - Get current status and stats
- `configure_optimizer(**kwargs)` - Configure optimization parameters
- `benchmark_optimization(...)` - Run performance benchmarks

### Classes

- `MetalKernelOptimizer` - Core optimization engine
- `AttentionConfig` - Attention pattern configuration
- `MLXLMIntegration` - Integration management
- `BenchmarkResult` - Benchmark result container

## 🤝 Contributing

1. Test on different model architectures
2. Optimize for specific sequence length ranges
3. Add support for new attention patterns
4. Improve Metal kernel performance
5. Add more comprehensive benchmarks

## 📜 License

This project is part of the OpenEvolve framework and follows the same licensing terms.

## 🙏 Acknowledgments

- Built on the AlphaEvolve framework for automated optimization discovery
- Inspired by the Metal kernel optimizations described in the AlphaEvolve paper
- Uses MLX and MLX-LM as the foundation for Apple Silicon machine learning
