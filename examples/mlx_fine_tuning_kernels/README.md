# MLX Fine-tuning Kernels - OpenEvolve Example

This example demonstrates optimizing **real fine-tuning operations** in MLX, inspired by [Liger Kernel's](https://github.com/linkedin/Liger-Kernel) proven optimizations. Instead of competing with MLX's highly optimized kernels, we create custom implementations of transformer operations that can be meaningfully improved over naive baselines.

## 🎯 The Real Opportunity

Liger Kernel demonstrated that **20%+ fine-tuning speedups** and **60% memory reductions** are achievable through optimized implementations of:
- **RMSNorm**: 3x speedup, 3x memory reduction
- **RoPE**: 3x speedup, 3x memory reduction  
- **SwiGLU**: 1.5x memory reduction
- **CrossEntropy**: 2x speedup, 4x memory reduction

This example targets **MLX equivalents** of these optimizations.

## 🚀 What Gets Optimized

### Core Transformer Operations

#### 1. **RMSNorm** - Layer Normalization
```python
# Baseline: Separate operations with forced evaluations
variance = mx.mean(x * x, axis=-1, keepdims=True)
mx.eval(variance)  # Inefficient!
rstd = mx.rsqrt(variance + eps)
mx.eval(rstd)
result = weight * (x * rstd)

# Optimization Target: Fused variance + rsqrt + scaling
# Expected: 2-3x speedup like Liger Kernel
```

#### 2. **RoPE** - Rotary Position Embeddings
```python
# Baseline: Multiple tensor operations, many intermediates
x1, x2 = x[..., ::2], x[..., 1::2]
# ... many temporary arrays and evaluations ...

# Optimization Target: Fused rotation computation
# Expected: 2-3x speedup
```

#### 3. **SwiGLU** - Gated Linear Unit
```python
# Baseline: Separate linear operations + activation
gate = mx.linear(x, w_gate)
gate_activated = mx.silu(gate)
up = mx.linear(x, w_up)
result = gate_activated * up

# Optimization Target: Fused linear + silu + multiply
# Expected: 50% memory reduction
```

#### 4. **CrossEntropy** - Loss Function
```python
# Baseline: Full logits materialization in memory
exp_logits = mx.exp(logits - max_logits)
# ... complete softmax for large vocabularies

# Optimization Target: Online/chunked computation
# Expected: 4x memory reduction
```

#### 5. **LoRA Linear** - Low-Rank Adaptation
```python
# Baseline: Separate base + LoRA computations
base_out = mx.linear(x, base_weight)
lora_out = mx.linear(mx.linear(x, lora_a), lora_b)

# Optimization Target: Fused LoRA computation
# Expected: Memory and speed improvements
```

## 📊 Two-Level Evaluation

### Level 1: Micro-benchmarks
Tests individual kernel performance against naive baselines:
- **Correctness**: Results must match baseline (< 1e-2 tolerance)
- **Speed**: Target 1.2x+ speedup per kernel
- **Memory**: Measure allocation efficiency

### Level 2: Real Model Macro-benchmark  
Tests **actual fine-tuning performance** using real HuggingFace MLX models:

#### **Comprehensive Real Model Testing**:
- **Multiple Real Models**: Tests across 2-5 actual MLX community models
  - `mlx-community/Qwen3-0.6B-bf16` (600M parameters)
  - `mlx-community/SmolLM-135M-Instruct-4bit` (135M parameters)  
  - `mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit` (1.1B parameters)
  - `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (500M parameters)
  - `mlx-community/Phi-3.5-mini-instruct-4bit` (3.8B parameters)

#### **Comprehensive Metrics**:
- **Training Speed**: Real fine-tuning speedup across models
- **Memory Efficiency**: VRAM usage improvements
- **Convergence Quality**: Loss trajectory analysis
- **Cross-Model Consistency**: Optimization robustness
- **NO SYNTHETIC MODELS**: Only real production models used

**This is the ultimate test** - do kernel optimizations provide consistent benefits across multiple real models that users actually fine-tune?

## 🏗️ Implementation Structure

### Evolved Kernels (`evolved_fine_tuning_kernels()`)
```python
# EVOLVE-BLOCK-START
def rms_norm(x, weight, eps=1e-6):
    # Custom RMSNorm with fusion opportunities
    # Target: 2-3x speedup vs naive baseline
    
def rope_embeddings(x, freqs_cos, freqs_sin):
    # Custom RoPE with optimized rotation
    # Target: 2-3x speedup vs naive baseline
    
def swiglu_activation(x, w_gate, w_up):
    # Custom SwiGLU with operation fusion  
    # Target: 50% memory reduction vs naive baseline
    
# ... other kernels
# EVOLVE-BLOCK-END
```

### Naive Baselines
Intentionally inefficient implementations with:
- Excessive `mx.eval()` calls (forces computation)
- Poor memory access patterns
- Missed fusion opportunities
- Many intermediate arrays

### Simple Transformer Model
Uses the custom kernels in a realistic transformer block for macro-benchmarking.

## 🎯 Expected Evolution Path

Based on Liger Kernel's proven optimizations:

1. **Early generations**: Remove unnecessary `mx.eval()` calls → 10-20% speedup
2. **Mid generations**: Fuse operations, optimize memory patterns → 20-40% speedup
3. **Later generations**: Mathematical simplifications, advanced fusion → 30-60% speedup

## 📈 Success Metrics

### Micro-benchmark Targets:
- **Minimum**: 1.2x average kernel speedup (20% improvement)
- **Good**: 1.5x average kernel speedup (50% improvement)  
- **Excellent**: 2.0x+ average kernel speedup (100%+ improvement)

### Macro-benchmark Targets:
- **Training speedup**: 20%+ faster fine-tuning to same loss
- **Memory reduction**: 30%+ lower peak memory usage
- **Correctness**: Same convergence quality

## 🚀 Usage

### Prerequisites
```bash
pip install mlx>=0.15.0 numpy psutil
# Or: pip install -r requirements.txt
```

### Optional: Enable Comprehensive Real Model Evaluation
For the most realistic benchmarks using multiple real HuggingFace models:
```bash
# Install comprehensive evaluation dependencies
python setup_comprehensive_evaluation.py

# Or manually:
pip install transformers>=4.35.0 mlx-lm>=0.3.0 datasets>=2.14.0 psutil
```

Comprehensive evaluation will test your kernels across multiple real models:
- `mlx-community/Qwen3-0.6B-bf16` (600M parameters) - Primary
- `mlx-community/SmolLM-135M-Instruct-4bit` (135M parameters) - Fast testing
- `mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit` (1.1B parameters) - Larger scale
- `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (500M parameters) - Alternative
- `mlx-community/Phi-3.5-mini-instruct-4bit` (3.8B parameters) - Large scale

**Benefits of comprehensive evaluation:**
- Tests across multiple model architectures and sizes
- Validates optimization consistency across real models
- Uses realistic instruction-following datasets
- Provides cross-model performance analysis
- NO synthetic model fallbacks

### Quick Test
```bash
cd examples/mlx_fine_tuning_kernels

# Test the initial implementation
python initial_program.py

# Test the evaluator
python evaluator.py
```

### Run Evolution
```bash
# Start optimization  
python ../../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

### Expected Output - Comprehensive Real Model Evaluation
```
🚀 Evaluating MLX Fine-tuning Kernels...

📊 MICRO-BENCHMARKS: Individual Kernel Performance
  rms_norm: 1.34x speedup, 0.85x memory (2.1ms vs 2.8ms) 🟢
  swiglu_activation: 1.41x speedup, 0.78x memory (3.2ms vs 4.5ms) 🟢
  … (all 6 kernels tested)

🚀 COMPREHENSIVE REAL MODEL EVALUATION
============================================================

🔍 Discovering available real models...
  Testing mlx-community/Qwen3-0.6B-bf16 (600M)...
    ✅ Tokenizer loaded
    ✅ Model available
  Testing mlx-community/SmolLM-135M-Instruct-4bit (135M)...
    ✅ Tokenizer loaded  
    ✅ Model available

📊 Found 2 available models:
  - mlx-community/Qwen3-0.6B-bf16 (600M)
  - mlx-community/SmolLM-135M-Instruct-4bit (135M)

🧪 Benchmarking mlx-community/Qwen3-0.6B-bf16 (600M)...
  Config: batch_size=2, seq_len=128, samples=200, epochs=5
    🔬 EVOLVED experiment...
      Generated 200 training samples
      Epoch 1/5: loss=2.1234, time=1.45s
      Epoch 5/5: loss=1.8765, time=1.23s
      EVOLVED completed: 6.85s total, 1.8765 final loss
    🔬 NAIVE experiment...
      Epoch 1/5: loss=2.1298, time=1.89s  
      Epoch 5/5: loss=1.8823, time=1.67s
      NAIVE completed: 8.92s total, 1.8823 final loss
  📊 Results: 1.30x speedup, 0.91x memory, 0.0058 loss diff

🧪 Benchmarking mlx-community/SmolLM-135M-Instruct-4bit (135M)...
  📊 Results: 1.38x speedup, 0.87x memory, 0.0076 loss diff

📊 COMPREHENSIVE RESULTS ACROSS 2 REAL MODELS:
  Models Tested: 600M, 135M
  Average Speedup: 1.34x
  Speedup Range: 1.30x - 1.38x
  Average Memory Ratio: 0.89x  
  Average Loss Difference: 0.0067
  Comprehensive Score: 0.745

🥇 VERY GOOD: Strong improvements on real models!

🏆 FINAL EVALUATION:
  Overall Score: 0.832
  Micro Score: 0.945
  Macro Score: 0.745  
  Real Models Tested: 2
  Cross-Model Consistency: High
🥈 EXCELLENT: Consistent strong performance across real models!
```

## 🏆 Why This Will Succeed

### ✅ **Proven Optimization Space**
- Liger Kernel demonstrates these optimizations work in practice
- Clear fusion opportunities in transformer operations  
- Realistic targets vs naive baselines (not competing with Apple's optimized kernels)

### ✅ **Real-World Validation**
- Tests actual fine-tuning performance, not just synthetic benchmarks
- Measures practical benefits: training speed and memory usage
- Uses realistic transformer architecture and operations

### ✅ **Appropriate Complexity**
- More meaningful than simple tensor operations
- Less complex than full Metal kernel programming
- Achievable through operation fusion and algorithmic improvements

### ✅ **Clear Success Metrics**
- **Binary correctness**: Pass/fail with reasonable tolerance
- **Primary metric**: Overall score combining micro + macro performance
- **Real impact**: Faster fine-tuning with less memory

## 🎓 Learning from AlphaEvolve Paper

This example applies AlphaEvolve's success principles correctly:

### ✅ **Right Problem Selection**
- **Paper**: Optimized existing algorithms (tiling heuristics)
- **This example**: Optimizes existing operations (transformer kernels)

### ✅ **Beatable Baseline**  
- **Paper**: Compared against existing solutions (improvable)
- **This example**: Compares against naive implementations (clearly improvable)

### ✅ **Clear Metrics**
- **Paper**: Direct performance measurement (kernel runtime)
- **This example**: Direct performance measurement (training speed + memory)

### ✅ **Incremental Improvement**
- **Paper**: 23% improvement through many optimizations  
- **This example**: Target 20-30% through step-by-step fusion

## 🔮 Real-World Impact

Success here would demonstrate:
- **MLX optimization capabilities**: Showing MLX can be optimized beyond naive implementations
- **Practical fine-tuning improvements**: Real speedups for the MLX community
- **OpenEvolve effectiveness**: Proving evolutionary optimization works on complex, practical problems

This represents a **genuinely valuable and achievable target** that bridges the gap between toy examples and production optimization challenges.

## 📚 References

- [Liger Kernel](https://github.com/linkedin/Liger-Kernel): Proven transformer optimizations for PyTorch
- [Unsloth](https://github.com/unslothai/unsloth): 2x faster training with custom kernels
- [AlphaEvolve Paper](https://arxiv.org/abs/2502.05229): Evolutionary optimization for coding problems
- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html): Apple's machine learning framework
