# MLX Training Performance Optimization with OpenEvolve

This example demonstrates using OpenEvolve to optimize MLX training performance on Apple Silicon, focusing exclusively on accelerating neural network training workloads.

## The Training-Focused Approach: Real-World MLX Training Optimization

We now focus exclusively on **MLX training performance** optimization:

✅ **Training Workloads**: Forward + backward passes with gradient computation  
✅ **Realistic Models**: Transformer architectures with substantial matrix operations  
✅ **Training Patterns**: Batch processing, MLP layers, attention computation  
✅ **Clear Signal**: Consistent evaluation without inference noise  
✅ **Practical Value**: Accelerate model development and research workflows  

## Why Training-Only Optimization?

### 1. **Cleaner Evaluation Signal**

Training provides much more consistent evaluation than inference:

```python
# Training: Deterministic, substantial computation
def training_step():
    inputs = mx.random.randint(0, vocab_size, (batch_size, seq_len))  # Fixed size
    logits = model(inputs)  # Deterministic forward pass
    loss, grads = mx.value_and_grad(loss_fn)(model, inputs, targets)  # Gradient computation
    optimizer.update(model, grads)  # Parameter updates
```

**Benefits:**
- No model loading overhead (1-2 second penalty eliminated)
- No text generation variability 
- Deterministic computation graphs
- Consistent matrix dimensions across runs
- More matrix operations per evaluation

### 2. **Training-Specific Matrix Patterns**

Training has unique characteristics that benefit from specialized optimization:

🧠 **Training Workload Patterns**:
- **Larger Batch Sizes**: 16-32 vs 1-4 for inference
- **Forward + Backward**: Double the matrix operations
- **Gradient Computation**: Requires transpose operations
- **Memory Pressure**: Activations + gradients + parameters
- **Repeated Patterns**: Same operations across many training steps

🎯 **Optimization Opportunities**:
- **Batch-Aware Tiling**: Different strategies for larger batch dimensions
- **Gradient-Friendly Patterns**: Consider transpose operations in backward pass
- **Memory Hierarchy**: Balance cache usage with gradient storage
- **Training Consistency**: Optimize for repeated execution patterns

### 3. **Substantial Practical Value**

Training optimization provides real benefits:
- **Faster Research Iteration**: Quicker model development cycles
- **Cost Reduction**: Lower compute costs for training runs  
- **Better Hardware Utilization**: More efficient use of Apple Silicon
- **Scalability**: Benefits increase with larger models and datasets

## Technical Implementation

### Matrix Operation Focus

The evolution targets the key functions used in training:

```python
def choose_tile_size(M, N, K, device_info):
    """
    Optimize for training-specific patterns:
    - Batch-heavy matrices (large M dimension)
    - MLP expansion/projection (4x hidden dimension scaling)
    - Attention computation (square-ish matrices)
    - Gradient computation (consider transpose patterns)
    """

def optimized_matmul(A, B, tile_M, tile_N, tile_K):
    """
    Implement tiled multiplication optimized for:
    - Training memory access patterns
    - Apple Silicon architecture
    - Cache efficiency with gradient storage
    """
```

### Enhanced Training Evaluation

The evaluator creates realistic training scenarios:

```python
class EnhancedTrainingModel(nn.Module):
    """
    Transformer-like model with substantial matrix operations:
    - Multiple MLP layers (4x expansion/projection)
    - Attention-like operations  
    - Large output projections
    - Forward + backward passes
    """

# Training Configuration
batch_size = 32      # Realistic training batch
seq_len = 512        # Longer sequences
hidden_dim = 1024    # Large hidden dimension
vocab_size = 6000    # Substantial vocabulary
```

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Training-Focused Optimization
```bash
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml --iterations 200
```

### Resume from Checkpoint
```bash
# If interrupted, resume with:
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml --checkpoint ./openevolve_output/mlx_training_optimization_db/checkpoints/checkpoint_XX --iterations 100
```

## Expected Results

The training-focused approach should discover optimizations providing:

📈 **Training Speedup**: 10-25% faster training steps  
🎯 **Consistent Optimization**: Better signal-to-noise ratio for evolution  
🔧 **Architecture-Aware**: M1/M2/M3/M4 specific optimizations  
⚡ **Memory Efficient**: Optimized for training's memory pressure  

## Evolution Discoveries

Based on training characteristics and Apple Silicon architecture, expect OpenEvolve to discover:

🧠 **Training Workload Classification**:
```python
is_batch_heavy = (M > 256)  # Large batch dimension
is_mlp = (aspect_ratio_K > 1.5)  # MLP 4x expansion patterns
is_gradient_computation = (transpose_pattern_detected)  # Backward pass
```

🔧 **Apple Silicon Training Optimization**:
```python
if "M4" in chip and is_batch_heavy:
    base_tile = 128; vector_align = 32  # Large tiles for AMX units
    memory_scale = 1.5  # Training can use more memory
elif is_mlp and training_workload:
    k_bias = 1.3  # Favor K dimension for MLP patterns
```

⚡ **Training Memory Patterns**:
```python
# Optimize for training's repeated execution
if total_elements > 1_000_000 and is_training:
    scale = 1.1  # Larger tiles for substantial computation
    cache_optimization = "training_friendly"  # Consider gradient storage
```

## Integration with Training Workflows

Once optimized, integrate with any MLX training code:

```python
import mlx.core as mx
from optimized_kernels import enable_training_optimizations

# Enable OpenEvolve training optimizations
enable_training_optimizations("./openevolve_output/best/best_program.py")

# Your existing training code gets automatic speedups!
for epoch in range(num_epochs):
    for batch in dataloader:
        loss, grads = mx.value_and_grad(loss_fn)(model, batch)
        optimizer.update(model, grads)  # Now faster!
```

## Comparison: Training vs Inference Optimization

| **Inference Optimization** | **Training Optimization** |
|------------------------------|---------------------------|
| ❌ Noisy evaluation (model loading, text generation) | ✅ Clean evaluation (deterministic computation) |
| ❌ Small matrices (batch=1-4) | ✅ Large matrices (batch=16-32) |
| ❌ Variable workloads | ✅ Consistent patterns |
| ❌ Complex pipeline overhead | ✅ Direct matrix operation focus |
| ❌ Difficult signal extraction | ✅ Clear optimization signal |

## Research Impact

This training-focused approach demonstrates:

1. **Practical AI Acceleration**: Directly optimizing the bottleneck of model development
2. **Hardware-Software Co-Design**: Training-specific optimizations for Apple Silicon  
3. **Clear Evaluation Methodology**: Robust metrics for evolutionary optimization
4. **Real-World Application**: Immediate benefits for ML researchers and practitioners

This moves from proof-of-concept to **production-ready training acceleration** that ML practitioners can immediately benefit from.
