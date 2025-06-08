# MLX LoRA Fine-tuning Optimization - OpenEvolve Example

This example demonstrates optimizing **real LoRA fine-tuning** using the official **MLX-LM library** by evolving kernels that can achieve the same training loss as the standard MLX-LM implementation but with improved memory efficiency and/or training speed.

## 🎯 The Real Challenge

Instead of optimizing theoretical kernels, this example targets **actual MLX-LM LoRA fine-tuning** optimization using the official mlx-lm library. The goal is to discover kernel implementations that can:

- **Achieve the same training loss** as standard MLX-LM LoRA fine-tuning
- **Reduce memory usage** during training
- **Increase training speed** (tokens/second)
- **Maintain numerical stability** and convergence quality
- **Use real MLX-LM infrastructure** for authentic benchmarking

This demonstrates real performance benefits like unsloth and liger kernel libraries provide for NVIDIA GPUs, but for MLX on Apple Silicon using production MLX-LM code.

## 🚀 What Gets Optimized

### Target Model & Dataset
- **Model**: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (500M parameters, 4-bit quantized)
- **Training**: Real LoRA fine-tuning using MLX-LM library on instruction-following dataset
- **Baseline**: Standard MLX-LM LoRA implementation (official mlx-lm code)
- **Metric**: Training loss convergence with efficiency improvements

### Core LoRA Operations for Optimization

#### 1. **LoRA Linear Forward Pass**
```python
# Standard MLX LoRA: Separate base + LoRA computation
base_out = x @ base_weight.T
lora_a_out = x @ lora_a.T  
lora_b_out = lora_a_out @ lora_b.T
result = base_out + scale * lora_b_out

# Optimization Target: Fused or pre-computed LoRA
# Expected: Memory reduction + speedup
```

#### 2. **LoRA Backward Pass & Gradient Computation**
```python
# Standard: Separate gradient computations for base, lora_a, lora_b
grad_base = grad_output @ x.T
grad_lora_b = grad_output @ lora_a_out.T  
grad_lora_a = lora_b.T @ grad_output @ x.T

# Optimization Target: Fused gradient computation
# Expected: Reduced memory allocations
```

#### 3. **Multi-Layer LoRA Application**
```python
# Standard: Apply LoRA to each layer separately (q_proj, v_proj, etc.)
for layer in model.layers:
    layer.self_attn.q_proj = LoRALinear.from_linear(layer.self_attn.q_proj)
    layer.self_attn.v_proj = LoRALinear.from_linear(layer.self_attn.v_proj)

# Optimization Target: Batch LoRA operations across layers
# Expected: Better memory utilization
```

#### 4. **Training Step Optimization**
```python
# Standard: Separate forward, loss, backward, optimizer steps
logits = model(inputs)
loss = cross_entropy(logits, targets)
grads = compute_gradients(loss)
optimizer.update(model, grads)

# Optimization Target: Fused training operations
# Expected: Reduced kernel launches and memory overhead
```

## 📊 Evaluation Approach

### Real LoRA Fine-tuning Benchmark
- **Model**: Uses actual MLX-LM models with standard architecture
- **Dataset**: Instruction-following examples (100 samples for quick testing)
- **Training**: 2 epochs, same hyperparameters for baseline and evolved
- **Metrics**: 
  - Training loss convergence (must match within 1% of baseline)
  - Training speed (tokens/second)
  - Peak memory usage (MB)
  - Memory efficiency (MB/token)

### Success Criteria
- **Primary**: Achieve same final training loss (±1%)
- **Secondary**: Memory reduction (10%+ improvement) OR speed improvement (10%+ improvement)
- **Ideal**: Both memory AND speed improvements

## 🏗️ Implementation Structure

### Official MLX-LM Integration
- Uses real MLX-LM models and training infrastructure (`mlx-community/Qwen2.5-0.5B-Instruct-4bit`)
- Leverages official MLX-LM functions: `linear_to_lora_layers`, `train`, `evaluate`, `load_dataset`
- Works with actual MLX-LM training pipelines and optimizers
- Uses MLX-LM's `TrainingArgs`, `CacheDataset`, and adapter saving mechanisms

### Evolved LoRA Kernels (`evolved_lora_kernels()`)
```python
# EVOLVE-BLOCK-START
def optimized_lora_fine_tuning(model_name, train_data_path, config, adapter_save_path):
    """Complete optimized LoRA fine-tuning pipeline using MLX-LM"""
    # Load model using official MLX-LM
    model, tokenizer = load(model_name)
    
    # Use MLX-LM dataset loading
    train_set, valid_set, test_set = load_dataset(args, tokenizer)
    
    # Apply LoRA using official functions with optimizations
    model.freeze()
    optimized_linear_to_lora_layers(model, num_layers, lora_parameters)
    
    # Optimized training loop using MLX-LM infrastructure
    optimized_training_loop(model, train_dataset, val_dataset, args, optimizer)
    
    # Evaluation using MLX-LM evaluate function
    final_loss = optimized_evaluate(model, test_dataset)

def optimized_linear_to_lora_layers(model, num_layers, lora_parameters):
    """Enhanced LoRA layer conversion based on mlx-lm's linear_to_lora_layers"""
    # Use official implementation with potential memory optimizations
    return linear_to_lora_layers(model, num_layers, lora_parameters)
# EVOLVE-BLOCK-END
```

### Realistic Baseline: Standard MLX-LM LoRA
- Uses official `linear_to_lora_layers()` from MLX-LM
- Standard MLX-LM training infrastructure with `train()` function
- Official MLX-LM dataset loading with `load_dataset()`
- Standard `TrainingArgs` and `CacheDataset` usage
- Works with real MLX-LM models and tokenizers

## 🎯 Expected Evolution Path

Based on proven LoRA optimization techniques:

1. **Early generations**: Reduce unnecessary memory allocations → 5-10% memory reduction
2. **Mid generations**: Fuse forward/backward operations → 10-15% speedup  
3. **Later generations**: Advanced mathematical optimizations → 20%+ improvements

## 📈 Success Metrics

### Training Convergence (Required):
- **Must achieve**: Same final training loss (±1% tolerance)
- **Must maintain**: Numerical stability and gradient flow

### Efficiency Improvements (Target):
- **Memory efficiency**: 10%+ reduction in peak memory usage
- **Training speed**: 10%+ improvement in tokens/second  
- **Ideal**: 15%+ improvement in both metrics

## 🚀 Usage

### Prerequisites
```bash
# Install MLX
pip install mlx>=0.15.0

# Install MLX-LM for real model support
pip install mlx-lm>=0.15.0

# Install other dependencies
pip install numpy psutil transformers

# Or install all at once:
pip install -r requirements.txt
```

### Quick Test
```bash
cd examples/mlx_fine_tuning_kernels

# Test the setup first
python test_setup.py

# Test the initial implementation
python initial_program.py

# Test real LoRA training evaluation
python evaluator.py
```

### Run Evolution
```bash
# Start optimization  
python ../../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

### Expected Output
```
🚀 Evaluating MLX-LM LoRA Fine-tuning Optimization...

✅ MLX-LM available for evaluation
✅ LoRA implementations loaded successfully

📊 MLX-LM LORA FINE-TUNING COMPARISON
  Model: mlx-community/Qwen2.5-0.5B-Instruct-4bit
  Trials: 1

--- Trial 1/1 ---
🔬 Testing BASELINE implementation...
Loading model: mlx-community/Qwen2.5-0.5B-Instruct-4bit
Loading datasets...
Applying baseline LoRA...
Trainable parameters: 2.097M
Total parameters: 494.033M
Starting baseline training...
  🧪 Running BASELINE LoRA fine-tuning...
    Final loss: 2.1234
    Training time: 12.45s
    Memory delta: 245.1 MB

🚀 Testing EVOLVED implementation...
Loading model: mlx-community/Qwen2.5-0.5B-Instruct-4bit
Loading datasets...
Applying LoRA...
Trainable parameters: 2.097M
Total parameters: 494.033M
Starting optimized training...
  🧪 Running EVOLVED LoRA fine-tuning...
    Final loss: 2.1189
    Training time: 10.82s
    Memory delta: 218.3 MB

📊 MLX-LM LORA FINE-TUNING OPTIMIZATION RESULTS:
  Loss Convergence: ✅ (diff: 0.0045)
  Speed Improvement: 1.15x
  Memory Improvement: 1.12x
  Time Improvement: 1.15x
  Convergence Score: 1.000
  Efficiency Score: 0.612
  Overall Score: 0.784

🥇 EXCELLENT: Strong improvements while maintaining convergence!
```

## 💡 Why This Will Succeed

### ✅ **Uses Real MLX Models**
- Integrates with actual MLX-LM models and architectures
- Tests on real model layers (attention projections, MLPs)
- Measures actual training metrics (loss, speed, memory)

### ✅ **Clear Success Metrics**
- **Binary convergence check**: Final loss must match (±1%)
- **Efficiency improvements**: Memory and/or speed gains
- **Real-world impact**: Actual fine-tuning becomes more efficient

### ✅ **Proven Optimization Space**
- LoRA operations have known optimization opportunities
- Weight pre-computation and fusion techniques
- Memory access pattern improvements
- Gradient computation optimization

### ✅ **Beatable Baseline**
- Standard MLX LoRA implementation (not heavily optimized)
- Room for kernel-level optimizations
- Opportunity for memory access pattern improvements

## 🎓 Learning from Production LoRA Optimizations

This example applies proven LoRA optimization techniques:

### ✅ **Weight Pre-computation**
- Pre-fuse LoRA weights when possible during inference
- Reduce matrix multiplications from 3 to 1

### ✅ **Memory-Efficient Gradients**  
- Optimize gradient computation patterns for LoRA structure
- Reduce intermediate tensor allocations

### ✅ **Training Loop Optimization**
- Fuse forward/backward/update operations
- Reduce kernel launch overhead

### ✅ **Multi-Layer Batch Processing**
- Apply LoRA optimizations across multiple layers efficiently
- Better utilize MLX's parallelization capabilities

## 🔮 Real-World Impact

Success here would demonstrate:
- **Practical LoRA optimization**: Real improvements for MLX fine-tuning
- **Production-ready techniques**: Optimizations that users can apply
- **OpenEvolve effectiveness**: Evolutionary approach works on realistic problems

This represents a **genuinely valuable optimization challenge** that bridges research and practical application in the MLX ecosystem, similar to how Unsloth provides 2x speedups and Liger Kernel provides 20%+ memory savings for NVIDIA GPUs.

## 📚 References

- [MLX-LM Documentation](https://github.com/ml-explore/mlx-examples): Apple's ML framework examples
- [LoRA Paper](https://arxiv.org/abs/2106.09685): Low-Rank Adaptation of Large Language Models
- [Unsloth](https://github.com/unslothai/unsloth): Proven LoRA speedup techniques for NVIDIA
- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html): Apple's ML framework
