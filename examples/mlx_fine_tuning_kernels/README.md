# MLX Quantized LoRA Fusion Optimization - OpenEvolve Example

This example demonstrates using OpenEvolve to discover optimized quantized LoRA kernels that eliminate the **dequantization bottleneck** in MLX-LM's LoRA implementation.

## 🎯 The Specific Problem

MLX-LM's current LoRA implementation has a critical inefficiency when working with quantized models:

```python
# From MLX-LM DoRALinear.__call__ - INEFFICIENT
def __call__(self, x):
    w = self._dequantized_weight()  # ❌ EXPENSIVE: Dequantizes entire weight matrix
    y = x @ w.T                     # ❌ Standard matmul on full-precision weights
    z = (self.dropout(x) @ self.lora_a) @ self.lora_b
    return y + (self.scale * z).astype(x.dtype)
```

**The Problem**: For quantized models (4-bit, 8-bit), MLX-LM dequantizes the entire base weight matrix just to perform the matrix multiplication, then discards the dequantized weights. This wastes memory and computation.

**The Opportunity**: MLX provides `mx.quantized_matmul()` which can perform matrix multiplication directly on quantized weights without dequantization.

## 🚀 The Optimization Target

OpenEvolve will discover optimized kernels that:

```python
# Target: EFFICIENT quantized LoRA computation
def optimized_call(self, x):
    # ✅ EFFICIENT: Direct quantized operations, no dequantization
    y = mx.quantized_matmul(x, self.quantized_weight, self.scales, self.biases,
                           group_size=self.group_size, bits=self.bits, transpose=True)
    z = efficient_lora_computation(x, self.lora_a, self.lora_b, self.scale)
    return y + z.astype(x.dtype)
```

## 📊 Expected Impact

Based on the inefficiency analysis, this optimization should achieve:

- **Memory Reduction**: 15-30% (by eliminating temporary dequantized weights)
- **Speed Improvement**: 10-20% (by using optimized quantized operations)
- **Same Accuracy**: Maintain identical training convergence and final loss
- **Broader Compatibility**: Work with all MLX quantized models (4-bit, 8-bit)

## 🔧 What Gets Optimized

### Core Target: OptimizedQuantizedLoRALinear Class

OpenEvolve will evolve the core LoRA computation to use MLX's quantized operations:

```python
# EVOLVE-BLOCK-START
class OptimizedQuantizedLoRALinear(nn.Module):
    def __call__(self, x):
        # EVOLUTION TARGET: Use mx.quantized_matmul directly
        base_out = mx.quantized_matmul(
            x, self.base_layer.weight, self.base_layer.scales, self.base_layer.biases,
            group_size=self.base_layer.group_size, bits=self.base_layer.bits, transpose=True
        )
        # Optimize LoRA computation patterns
        lora_out = optimized_lora_computation(x, self.lora_a, self.lora_b, self.scale)
        return base_out + lora_out.astype(base_out.dtype)
# EVOLVE-BLOCK-END
```

### Secondary Targets:

1. **Compiled Quantized Operations**: Using `@mx.compile` for quantized LoRA fusion
2. **Memory-Efficient Patterns**: Strategic cache clearing and memory management
3. **Apple Silicon Optimization**: Unified memory architecture optimizations

## 🧪 Evaluation Approach

### Test Model
- **Model**: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (quantized)
- **Task**: Instruction-following fine-tuning
- **Baseline**: Standard MLX-LM quantized LoRA
- **Metric**: Memory usage, training speed, numerical accuracy

### Success Criteria
- **Primary**: Same final training loss (±1% tolerance)
- **Secondary**: Memory reduction AND/OR speed improvement
- **Target**: 15%+ efficiency gain while maintaining accuracy

### Evaluation Process
1. **Baseline Measurement**: Standard MLX-LM quantized LoRA performance
2. **Evolved Measurement**: Optimized quantized LoRA kernels performance
3. **Comparison**: Memory, speed, and accuracy analysis

## 🏗️ Implementation Structure

### Real MLX-LM Integration
- Uses actual quantized MLX-LM models (`mlx-community/Qwen2.5-0.5B-Instruct-4bit`)
- Integrates with MLX-LM training infrastructure
- Measures real memory usage and training performance
- Maintains compatibility with MLX-LM LoRA APIs

### Evolution Focus Areas

1. **Quantized Matrix Operations**:
   ```python
   # Target: Replace dequantization with direct quantized ops
   mx.quantized_matmul(x, quantized_weight, scales, biases, group_size, bits, transpose=True)
   ```

2. **LoRA Computation Fusion**:
   ```python
   # Target: Efficient LoRA matrix multiplication patterns
   @mx.compile
   def optimized_lora_matmul(x, lora_a, lora_b, scale):
       return scale * mx.matmul(mx.matmul(x, lora_a), lora_b)
   ```

3. **Memory Management**:
   ```python
   # Target: Apple Silicon-optimized memory patterns
   def quantized_model_memory_optimizer(model):
       # Optimize memory limits for quantized models
   ```

## 🎯 Why This Will Succeed

### ✅ **Clear Inefficiency Target**
- Specific bottleneck: unnecessary dequantization in LoRA forward pass
- Measurable impact: memory usage and training speed
- Available solution: `mx.quantized_matmul()` exists and works

### ✅ **Realistic Optimization Scope**
- Algorithm-level optimization, not low-level kernel development
- Uses existing MLX primitives in more efficient patterns
- Similar to proven optimizations (Unsloth, Liger Kernels)

### ✅ **Concrete Success Metrics**
- Binary convergence check: final loss must match (±1%)
- Memory efficiency: measurable reduction in peak memory usage
- Speed improvement: measurable training time reduction

### ✅ **Proven Optimization Pattern**
This follows the same pattern as successful optimizations:
- **Unsloth**: 2x LoRA speedup by avoiding unnecessary operations
- **Liger Kernels**: 20% memory savings through operation fusion
- **AlphaEvolve**: Kernel optimizations discovered through automated search

## 🚀 Usage

### Prerequisites
```bash
# Install MLX and MLX-LM
pip install mlx>=0.15.0 mlx-lm>=0.15.0

# Install dependencies
pip install -r requirements.txt
```

### Quick Test
```bash
cd examples/mlx_fine_tuning_kernels

# Test the quantized optimization setup
python initial_program.py

# Test the evaluator
python evaluator.py
```

### Run Evolution
```bash
# Start quantized LoRA optimization evolution
python ../../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

### Expected Output
```
🚀 Evaluating MLX Quantized LoRA Optimization...

📊 QUANTIZED LORA OPTIMIZATION BENCHMARK
  Model: mlx-community/Qwen2.5-0.5B-Instruct-4bit
  Target: Quantized LoRA fusion optimization

🔬 PHASE 1: Running BASELINE trials (standard quantized LoRA)
  🧪 Running BASELINE-1...
    Final loss: 1.234
    Training time: 15.2s
    Memory delta: 180.5 MB
    Peak memory delta: 220.3 MB

🚀 PHASE 2: Running EVOLVED trials (optimized quantized LoRA)
  🧪 Running EVOLVED-1...
    Final loss: 1.236
    Training time: 12.8s
    Memory delta: 145.2 MB
    Peak memory delta: 175.1 MB

📊 QUANTIZED LORA OPTIMIZATION RESULTS:
  Loss Convergence: ✅ (diff: 0.002)
  Speed Improvement: 1.19x
  Memory Improvement: 1.24x
  Peak Memory Improvement: 1.26x
  Overall Score: 0.785

🥇 EXCELLENT: Strong quantized LoRA optimizations achieved!
```

## 💡 Technical Innovation

This example represents a **concrete, achievable optimization** that:

### **Targets Real Inefficiency**
- MLX-LM actually dequantizes weights unnecessarily
- `mx.quantized_matmul()` provides the solution
- Measurable performance impact

### **Uses Algorithmic Optimization**
- Works at the mathematical operation level
- Uses existing MLX primitives more efficiently
- Doesn't require new kernel development

### **Provides Immediate Value**
- Applicable to all quantized MLX models
- Benefits any LoRA fine-tuning workflow
- Maintains full compatibility with MLX-LM

## 🔮 Real-World Impact

Success here demonstrates:
- **Practical Optimization**: Real memory and speed improvements for MLX users
- **OpenEvolve Effectiveness**: Automated discovery of concrete optimizations
- **MLX Ecosystem Value**: Contributions to Apple's ML framework

This represents a **genuinely valuable optimization** that could be contributed back to the MLX-LM project, providing real benefits to the Apple Silicon ML community.

## 📚 References

- [MLX Documentation](https://ml-explore.github.io/mlx/): Apple's ML framework
- [MLX-LM Repository](https://github.com/ml-explore/mlx-examples): Official MLX language models
- [Quantized Operations in MLX](https://ml-explore.github.io/mlx/build/html/python/mlx.core.html#mlx.core.quantized_matmul): MLX quantized matrix operations
- [LoRA Paper](https://arxiv.org/abs/2106.09685): Low-Rank Adaptation technique
- [Unsloth](https://github.com/unslothai/unsloth): Proven LoRA optimizations for reference
