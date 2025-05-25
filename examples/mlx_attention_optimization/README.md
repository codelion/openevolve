# MLX Attention Optimization Example

This example implements **High-Level ML Kernel Optimization** inspired by AlphaEvolve's **Gemini kernel engineering** approach (Section 3.3.2), but adapted for **realistic Python/MLX optimization** on Apple Silicon.

## 🎯 Why Attention Optimization?

Unlike low-level matrix multiplication (where MLX's C++/Metal kernels are hard to beat), **attention mechanisms** offer genuine opportunities for optimization at the algorithm level:

- **Complex multi-step operations** with room for fusion and reordering
- **Memory access patterns** that can be optimized for Apple Silicon's unified memory
- **Numerical precision tradeoffs** that affect both speed and accuracy
- **Sequence length handling** strategies for different workloads
- **Multi-head computation** patterns that can be optimized

## 🔬 What We're Optimizing

### **Core Attention Parameters (Evolvable)**
```python
def get_attention_config():
    return {
        "attention_dtype": "float32",        # ← float32/float16/bfloat16
        "memory_layout": "standard",         # ← standard/transposed/blocked  
        "chunking_strategy": "none",         # ← none/query_chunks/key_chunks/both
        "chunk_size": 512,                   # ← 128/256/512/1024
        "softmax_precision": "high",         # ← high/medium/fast
        "scale_strategy": "sqrt_dk",         # ← sqrt_dk/learned/fixed
        "use_fused_qkv": True,              # ← fusion optimizations
        "kv_cache_optimized": False         # ← inference optimizations
    }
```

### **Optimization Strategies**
1. **Memory Layout Optimization**: How Q, K, V matrices are arranged in memory
2. **Precision Strategies**: When to use float16 vs float32 for speed/accuracy balance
3. **Chunking Algorithms**: Breaking large sequences into cache-friendly chunks
4. **Fused Operations**: Combining multiple attention steps to reduce memory bandwidth
5. **Computation Ordering**: Optimizing the sequence of operations for Apple Silicon

## 🏗️ Architecture

### **Initial Implementation (`initial_program.py`)**
- **Comprehensive attention kernel** with multiple optimization strategies
- **Configurable parameters** for all major attention optimizations
- **Memory layout options** (standard, transposed, blocked)
- **Chunking strategies** for long sequences
- **Precision control** for speed/accuracy tradeoffs

### **Evaluation Framework (`evaluator.py`)**
- **Correctness verification** against reference MLX attention
- **Performance benchmarking** on realistic model configurations
- **Full model inference testing** using simplified transformer blocks
- **Multi-objective optimization**: speed + accuracy + memory efficiency

### **Test Configurations**
Based on models like **Qwen3-0.6B-bf16**:
- **Batch sizes**: 1, 2, 4, 8 (typical inference/training)
- **Sequence lengths**: 128, 256, 512, 1024, 2048
- **Model dimensions**: 256, 512, 768, 1024 (small to medium models)
- **Number of heads**: 8, 12, 16

## 📊 Expected Results

### **Realistic Performance Targets**
Based on attention complexity, we expect:
- **10-30% speedup** over standard MLX attention (realistic for Python optimization)
- **Memory efficiency gains** through better chunking and layout
- **Accuracy preservation** (numerical error < 1e-3)
- **Robust performance** across different model sizes

### **Key Optimizations We Expect Evolution to Discover**
1. **Float16 strategies** where accuracy allows (~20-30% speedup potential)
2. **Optimal chunk sizes** for Apple Silicon memory hierarchy (likely 256-512)
3. **Memory layout patterns** optimized for unified memory architecture
4. **Fused operation sequences** to reduce memory bandwidth
5. **Precision mixing** (high precision for critical steps, lower for others)

## 🚀 Running the Example

### **Prerequisites**
```bash
# Install MLX (Apple Silicon only)
pip install mlx

# Ensure OpenEvolve is installed
pip install -e .
```

### **Quick Test**
Verify the setup works:
```bash
cd examples/mlx_attention_optimization
python initial_program.py
```

Expected output:
```
MLX Attention Optimization Example
Current configuration: {'attention_dtype': 'float32', 'memory_layout': 'standard', ...}

Running benchmark...
Results:
  b1_s128_d256: 0.0045s, 12.34 GFLOPS
  b1_s512_d512: 0.0234s, 23.45 GFLOPS
  ...
```

### **Run Evolution**
```bash
# Quick test (50 iterations, ~30 minutes)
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml --iterations 50

# Standard run (150 iterations, ~2-3 hours) 
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml --iterations 150

# Full optimization (300 iterations, ~6-8 hours)
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml --iterations 300
```

## 📈 Understanding the Results

### **Key Metrics**
- **`attention_efficiency`**: Primary optimization target (0-1 scale)
- **`model_efficiency`**: Speedup on full model inference (>1.0 is good)
- **`correctness_score`**: Numerical accuracy vs reference (should be ~1.0)
- **`avg_speedup`**: Average speedup across all model configurations
- **`avg_throughput_gflops`**: Raw attention throughput

### **Success Indicators**
- **Model efficiency > 1.1**: 10%+ speedup on real model inference
- **Correctness score > 0.99**: Maintains numerical accuracy
- **Attention efficiency > 0.7**: Good overall optimization

### **Evolution Progress**
```
INFO - Iteration 75: Child abc123 from parent def456 in 45.67s.
Metrics: attention_efficiency=0.7234, model_efficiency=1.1456, correctness_score=0.9987
(Δ: attention_efficiency=+0.0234, model_efficiency=+0.0456)
```

## 🔍 Comparison to AlphaEvolve Paper

| **Aspect** | **AlphaEvolve (TPU)** | **Our Implementation (MLX)** |
|------------|----------------------|------------------------------|
| **Target** | Pallas kernel tiling | Attention algorithm optimization |
| **Hardware** | Google TPU | Apple Silicon GPU |
| **Scope** | Low-level kernel parameters | High-level algorithm strategies |
| **Language** | TPU assembly/Pallas | Python/MLX |
| **Optimization Space** | Tile shapes, memory patterns | Attention fusion, precision, chunking |
| **Expected Improvement** | 23% kernel speedup | 10-30% attention speedup |
| **Evaluation** | Real TPU performance | Real model inference on Apple Silicon |

## 🎯 Why This Approach Works

### **Realistic Optimization Scope**
- **Algorithm-level optimizations** rather than competing with optimized C++ kernels
- **Memory access pattern improvements** for Apple Silicon's architecture
- **Numerical precision strategies** that balance speed and accuracy
- **Computation fusion** at the Python/MLX level

### **Genuine Room for Improvement**
- **Standard MLX attention** is not necessarily optimized for all use cases
- **Memory layout choices** can significantly impact performance
- **Precision strategies** offer real speed/accuracy tradeoffs
- **Chunking algorithms** can improve memory efficiency for long sequences

### **Measurable Real-World Impact**
- **Full model inference testing** ensures practical relevance
- **Multiple model configurations** validate generalization
- **Correctness verification** ensures reliability
- **Performance comparison** provides clear improvement metrics

## 🔬 Advanced Usage

### **Custom Model Testing**
Modify `evaluator.py` to test on your specific model:
```python
# Add your model configuration
model_configs = [
    {"d_model": your_d_model, "n_heads": your_n_heads, "n_layers": 2, "seq_len": your_seq_len}
]
```

### **Production Integration**
Use evolved configurations in real models:
```python
# Load best configuration
with open("openevolve_output/best/best_program_info.json") as f:
    best_config = json.load(f)["metrics"]

# Apply to your model
optimized_attention = partial(optimized_attention_kernel, **best_config)
```

### **Comparative Analysis**
Compare different optimization strategies:
```python
# Test float16 vs float32
config_fp16 = {"attention_dtype": "float16", ...}
config_fp32 = {"attention_dtype": "float32", ...}
```

## 🎓 Learning Outcomes

This example demonstrates:
- **Realistic scope** for Python-based ML optimization
- **Multi-objective optimization** balancing speed, accuracy, and memory
- **Real-world evaluation** on transformer model inference
- **Evolutionary discovery** of non-obvious optimization strategies

Unlike the matrix multiplication example, this has genuine potential to discover optimizations that outperform naive implementations while remaining practically implementable.

## 🔧 Troubleshooting

**Common Issues:**
- **MLX import errors**: Ensure you're on Apple Silicon and MLX is installed
- **Memory errors**: Reduce batch sizes or sequence lengths in config
- **Slow evaluation**: Reduce the number of test configurations
- **Correctness failures**: Check tolerance values in evaluator

**Performance Tips:**
- **Monitor memory usage** during evolution
- **Start with shorter sequences** for faster iteration
- **Use checkpointing** for long evolution runs
- **Analyze intermediate results** to understand optimization trends

This example represents a more realistic and achievable optimization target compared to competing with highly optimized BLAS libraries, while still demonstrating the power of evolutionary code optimization for real ML workloads.
