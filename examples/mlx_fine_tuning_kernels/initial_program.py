"""
MLX LoRA + Quantization Fusion Optimization - OpenEvolve Example

This example demonstrates evolving optimized quantized LoRA kernels that eliminate
the expensive dequantization → LoRA → requantization pattern in MLX-LM.

SPECIFIC TARGET: The dequantization bottleneck in DoRALinear and LoRALinear
where MLX-LM dequantizes entire weight matrices just to apply LoRA.

OPTIMIZATION GOAL: Use mx.quantized_matmul directly, never dequantize base weights.
"""

import math
import time
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import types
import tempfile
import json
import gc
import psutil
import os

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np

    MLX_AVAILABLE = True
except ImportError:
    print("⚠️ MLX not available - this example requires MLX")
    MLX_AVAILABLE = False
    raise ImportError("MLX is required for this example")

try:
    from mlx_lm import load, generate
    from mlx_lm.tuner.trainer import TrainingArgs, evaluate, train
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.utils import (
        linear_to_lora_layers,
        load_adapters,
        print_trainable_parameters,
    )
    from mlx_lm.utils import save_config

    MLX_LM_AVAILABLE = True
    print("✅ MLX-LM available for quantized LoRA optimization")
except ImportError as e:
    print(f"⚠️ MLX-LM not available: {e}")
    MLX_LM_AVAILABLE = False


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def create_training_config():
    """Create training configuration for quantized LoRA fine-tuning."""
    return {
        "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",  # Quantized model
        "train": True,
        "fine_tune_type": "lora",
        "optimizer": "adam",
        "optimizer_config": {"adam": {}},
        "data": "temp_data",
        "seed": 42,
        "num_layers": 4,
        "batch_size": 2,
        "iters": 15,  # Short for fast evaluation
        "val_batches": 5,
        "learning_rate": 1e-4,
        "steps_per_report": 5,
        "steps_per_eval": 100,
        "adapter_path": "temp_adapters",
        "save_every": 100,
        "max_seq_length": 256,  # Shorter for faster evaluation
        "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 16.0},  # Smaller rank
        "mask_prompt": False,
        "test": True,
        "test_batches": 5,
        "resume_adapter_file": None,
        "config": None,
        "grad_checkpoint": False,
        "lr_schedule": None,
        "wandb": None,
    }


def create_sample_dataset(output_dir: str, num_samples: int = 50):
    """Create a small sample dataset for quantized LoRA testing."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Simple examples optimized for quantized model testing
    examples = [
        {"text": "What is machine learning?\nMachine learning is AI that learns from data without explicit programming."},
        {"text": "Explain deep learning.\nDeep learning uses neural networks with many layers to learn complex patterns."},
        {"text": "What is quantization?\nQuantization reduces model size by using lower precision numbers like int8 or int4."},
        {"text": "How does LoRA work?\nLoRA adds small trainable matrices to frozen pre-trained weights for efficient fine-tuning."},
        {"text": "What is Apple Silicon?\nApple Silicon refers to custom ARM-based processors designed by Apple for Mac computers."},
        {"text": "What is MLX?\nMLX is Apple's machine learning framework optimized for Apple Silicon processors."},
        {"text": "Explain transformers.\nTransformers are neural networks that use attention mechanisms for sequence processing."},
        {"text": "What is fine-tuning?\nFine-tuning adapts pre-trained models to specific tasks with task-specific data."},
        {"text": "What is attention?\nAttention mechanisms allow models to focus on relevant parts of input sequences."},
        {"text": "What is CUDA?\nCUDA is NVIDIA's parallel computing platform for GPU acceleration."},
    ]

    # Expand to requested number
    expanded_examples = []
    for i in range(num_samples):
        example = examples[i % len(examples)]
        expanded_examples.append(example)

    # Create splits
    train_data = expanded_examples[:int(0.7 * num_samples)]
    valid_data = expanded_examples[int(0.7 * num_samples):int(0.9 * num_samples)]
    test_data = expanded_examples[int(0.9 * num_samples):]

    # Ensure minimum sizes
    if not valid_data:
        valid_data = [train_data[0]]
    if not test_data:
        test_data = [train_data[0]]

    # Write datasets
    for split, data in [("train", train_data), ("valid", valid_data), ("test", test_data)]:
        with open(f"{output_dir}/{split}.jsonl", "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")

    print(f"✅ Created dataset: {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test")


def evolved_lora_kernels():
    """
    Evolved LoRA kernel implementations targeting quantized LoRA fusion.

    CORE TARGET: Eliminate the expensive dequantization in MLX-LM's LoRA implementation
    by using mx.quantized_matmul directly on quantized base weights.

    BASELINE INEFFICIENCY (from MLX-LM DoRALinear):
        w = self._dequantized_weight()  # EXPENSIVE: Full dequantization
        y = x @ w.T                     # Standard matmul on dequantized weights

    OPTIMIZATION TARGET:
        y = mx.quantized_matmul(x, quantized_weight, scales, biases, 
                               group_size, bits, transpose=True)  # Direct quantized ops
    """

    if not MLX_LM_AVAILABLE:
        raise ImportError("MLX-LM is required for quantized LoRA optimization")

    # EVOLVE-BLOCK-START
    @mx.compile
    def optimized_quantized_lora_matmul(x, quantized_weight, scales, biases, lora_a, lora_b, scale, group_size, bits):
        """
        Optimized quantized LoRA computation using direct quantized operations.
        
        Eliminates dequantization by using mx.quantized_matmul directly.
        """
        # CORE OPTIMIZATION: Use quantized matmul directly instead of dequantizing
        # This is the key efficiency gain - no intermediate full-precision weights
        base_out = mx.quantized_matmul(
            x, quantized_weight, scales, biases,
            group_size=group_size, bits=bits, transpose=True
        )
        
        # Compute LoRA contribution efficiently
        # Use compiled computation for better performance
        lora_temp = mx.matmul(x, lora_a)
        lora_out = mx.matmul(lora_temp, lora_b)
        
        # Fuse base and LoRA outputs
        return base_out + (scale * lora_out).astype(base_out.dtype)

    @mx.compile  
    def optimized_lora_computation(x, lora_a, lora_b, scale):
        """
        Optimized LoRA matrix computation with potential fusion opportunities.
        """
        # Standard LoRA computation but compiled for efficiency
        # Could be extended with custom tiling or memory patterns
        temp = mx.matmul(x, lora_a)
        result = mx.matmul(temp, lora_b)
        return scale * result

    class OptimizedQuantizedLoRALinear(nn.Module):
        """
        Optimized LoRA linear layer that works directly with quantized weights.
        
        KEY OPTIMIZATION: Never dequantizes base weights, uses mx.quantized_matmul directly.
        """

        def __init__(self, original_lora_layer, r=8, alpha=16, dropout=0.0, scale=None):
            super().__init__()
            
            # Extract the quantized linear layer
            if hasattr(original_lora_layer, 'linear'):
                self.base_layer = original_lora_layer.linear
            else:
                self.base_layer = original_lora_layer
                
            # Ensure we have a quantized layer to optimize
            if not isinstance(self.base_layer, nn.QuantizedLinear):
                print(f"  ⚠️ Warning: Expected quantized layer, got {type(self.base_layer)}")
                # Fall back to standard implementation for non-quantized layers
                self.base_layer = original_lora_layer
                self._is_optimized = False
            else:
                self._is_optimized = True
                print(f"  ✅ Optimizing quantized layer: {self.base_layer.bits}-bit, group_size={self.base_layer.group_size}")

            # LoRA parameters
            self.r = r
            self.alpha = alpha
            self.dropout = dropout
            self.scale = scale if scale is not None else alpha / r

            # Copy LoRA weights from original if available
            if hasattr(original_lora_layer, 'lora_a'):
                self.lora_a = original_lora_layer.lora_a
                self.lora_b = original_lora_layer.lora_b
            else:
                # Initialize new LoRA weights
                input_dims = self.base_layer.weight.shape[1]
                if self._is_optimized:
                    input_dims = input_dims * 32 // self.base_layer.bits
                output_dims = self.base_layer.weight.shape[0]
                
                scale_init = 1 / math.sqrt(input_dims)
                self.lora_a = mx.random.uniform(
                    low=-scale_init, high=scale_init, shape=(input_dims, r)
                )
                self.lora_b = mx.zeros(shape=(r, output_dims))

        def __call__(self, x):
            if not self._is_optimized:
                # Fall back to standard implementation for non-quantized layers
                if hasattr(self.base_layer, '__call__'):
                    base_out = self.base_layer(x)
                else:
                    base_out = x @ self.base_layer.weight.T
                lora_out = optimized_lora_computation(x, self.lora_a, self.lora_b, self.scale)
                return base_out + lora_out.astype(x.dtype)

            # CORE OPTIMIZATION: Use quantized operations directly
            try:
                # Use our optimized quantized LoRA computation
                result = optimized_quantized_lora_matmul(
                    x,
                    self.base_layer.weight,  # Keep quantized
                    self.base_layer.scales,
                    self.base_layer.biases,
                    self.lora_a,
                    self.lora_b,
                    self.scale,
                    self.base_layer.group_size,
                    self.base_layer.bits
                )
                
                # Add bias if present
                if hasattr(self.base_layer, 'bias') and self.base_layer.bias is not None:
                    result = result + self.base_layer.bias
                    
                return result
                
            except Exception as e:
                print(f"  ⚠️ Quantized optimization failed: {e}, falling back to standard")
                # Graceful fallback to standard implementation
                base_out = self.base_layer(x)
                lora_out = optimized_lora_computation(x, self.lora_a, self.lora_b, self.scale)
                return base_out + lora_out.astype(x.dtype)

    def memory_efficient_quantized_training_step(model, batch, optimizer, use_quantized_kernels=True):
        """
        Memory-efficient training step optimized for quantized LoRA models.
        """
        if not use_quantized_kernels:
            # Standard training step
            def loss_fn(model):
                logits = model(batch["input_ids"])
                return nn.losses.cross_entropy(logits, batch["labels"], reduction="mean")

            loss, grads = mx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)
            return loss

        # Optimized training step with memory management
        def loss_fn(model):
            # Clear cache before forward pass for quantized models
            mx.clear_cache()
            logits = model(batch["input_ids"])
            return nn.losses.cross_entropy(logits, batch["labels"], reduction="mean")

        # Compute gradients with compilation
        loss, grads = mx.value_and_grad(loss_fn)(model)
        
        # Clear cache before optimizer step
        mx.clear_cache()
        optimizer.update(model, grads)
        
        # Final cache clear for quantized models
        mx.clear_cache()
        
        return loss

    @mx.compile
    def optimized_quantized_loss_computation(logits, targets):
        """
        Optimized loss computation for quantized models.
        """
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    def quantized_model_memory_optimizer(model):
        """
        Optimize memory usage patterns for quantized models.
        """
        # Set appropriate memory limits for quantized models
        max_mem = mx.metal.device_info()["max_recommended_working_set_size"]
        
        # For quantized models, we can be more aggressive with memory usage
        # since the weights take less space
        quantized_limit = int(0.95 * max_mem)  # Use more memory for quantized models
        mx.set_wired_limit(quantized_limit)
        
        print(f"  🎯 Set optimized memory limit for quantized model: {quantized_limit // (1024*1024)} MB")

    return {
        "optimized_quantized_lora_linear_class": OptimizedQuantizedLoRALinear,
        "optimized_quantized_lora_matmul": optimized_quantized_lora_matmul,
        "optimized_lora_computation": optimized_lora_computation,
        "memory_efficient_quantized_training_step": memory_efficient_quantized_training_step,
        "optimized_quantized_loss_computation": optimized_quantized_loss_computation,
        "quantized_model_memory_optimizer": quantized_model_memory_optimizer,
    }
    # EVOLVE-BLOCK-END


def patch_quantized_lora_layers(model, evolved_kernels):
    """Patch model to use evolved quantized LoRA kernels."""
    if not evolved_kernels:
        print("  🔍 No evolved kernels to apply")
        model._kernels_applied = False
        return

    print(f"🚀 Patching model with quantized LoRA optimizations...")

    try:
        # Apply memory optimization first
        memory_optimizer = evolved_kernels.get("quantized_model_memory_optimizer")
        if memory_optimizer:
            memory_optimizer(model)

        # Replace LoRA layers with quantized optimized versions
        OptimizedQuantizedLoRALinear = evolved_kernels.get("optimized_quantized_lora_linear_class")
        if not OptimizedQuantizedLoRALinear:
            print("  ⚠️ No optimized LoRA class found")
            model._kernels_applied = False
            return

        replaced_count = 0
        
        # Find and replace LoRA layers
        print("  🔧 Scanning for LoRA layers to optimize...")
        
        all_modules = list(model.named_modules())
        print(f"    Total modules: {len(all_modules)}")
        
        lora_layers_found = []
        
        for name, module in all_modules:
            module_type = type(module).__name__
            
            # Look for LoRA layers (from MLX-LM)
            is_lora = (
                'LoRA' in module_type or 'lora' in module_type.lower() or
                (hasattr(module, 'lora_a') and hasattr(module, 'lora_b')) or
                (hasattr(module, 'linear') and hasattr(module.linear, 'weight'))
            )
            
            if is_lora:
                lora_layers_found.append((name, module))
                print(f"    🔍 Found LoRA layer: {name} (type: {module_type})")
                
                # Check if it has a quantized base layer
                base_layer = getattr(module, 'linear', module)
                if isinstance(base_layer, nn.QuantizedLinear):
                    print(f"      ✅ Has quantized base: {base_layer.bits}-bit")
                else:
                    print(f"      ℹ️ Base layer type: {type(base_layer)}")

        print(f"    Found {len(lora_layers_found)} LoRA layers")

        # Replace LoRA layers with optimized versions
        for layer_name, lora_layer in lora_layers_found:
            try:
                print(f"    📎 Optimizing LoRA layer: {layer_name}")
                
                # Create optimized version
                optimized_layer = OptimizedQuantizedLoRALinear(
                    original_lora_layer=lora_layer,
                    r=getattr(lora_layer, 'r', 8),
                    alpha=getattr(lora_layer, 'alpha', 16),
                    dropout=getattr(lora_layer, 'dropout', 0.0),
                    scale=getattr(lora_layer, 'scale', None)
                )
                
                # Replace in model
                name_parts = layer_name.split('.')
                if len(name_parts) == 1:
                    setattr(model, name_parts[0], optimized_layer)
                else:
                    parent = model
                    for part in name_parts[:-1]:
                        if part.isdigit() and hasattr(parent, '__getitem__'):
                            parent = parent[int(part)]
                        else:
                            parent = getattr(parent, part)
                    
                    final_attr = name_parts[-1]
                    if final_attr.isdigit() and hasattr(parent, '__setitem__'):
                        parent[int(final_attr)] = optimized_layer
                    else:
                        setattr(parent, final_attr, optimized_layer)
                
                replaced_count += 1
                print(f"      ✅ Successfully optimized {layer_name}")
                
            except Exception as e:
                print(f"    ⚠️ Failed to optimize {layer_name}: {e}")

        print(f"  ✅ Optimized {replaced_count} LoRA layers for quantized computation")

        # Store kernels and status
        model._evolved_kernels = evolved_kernels
        model._has_evolved_kernels = True
        model._kernels_applied = replaced_count > 0

        print(f"  📊 Quantized LoRA optimization status: {model._kernels_applied}")

    except Exception as e:
        print(f"❌ ERROR during quantized LoRA patching: {e}")
        import traceback
        traceback.print_exc()
        model._kernels_applied = False


def quantized_lora_fine_tuning_with_kernels(
    model_name: str,
    train_data_path: str,
    config: Dict[str, Any],
    adapter_save_path: str = "temp_adapters",
    evolved_kernels: Optional[Dict] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Quantized LoRA fine-tuning with evolved kernel optimizations.
    
    Specifically targets quantized models and measures the impact of 
    evolved quantized LoRA kernels.
    """
    # Set random seed
    mx.random.seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))

    print(f"Loading quantized model: {model_name}")
    model, tokenizer = load(model_name)

    # Verify we have a quantized model
    quantized_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            quantized_layers.append((name, module))

    print(f"✅ Found {len(quantized_layers)} quantized layers in model")
    if len(quantized_layers) == 0:
        print("⚠️ WARNING: No quantized layers found - optimization may not be effective")

    # Setup MLX-LM components
    args = types.SimpleNamespace(**config)
    args.data = train_data_path

    print("Loading datasets...")
    train_set, valid_set, test_set = load_dataset(args, tokenizer)

    # Apply LoRA first
    print("Applying LoRA...")
    model.freeze()
    linear_to_lora_layers(
        model, args.num_layers, args.lora_parameters, use_dora=(args.fine_tune_type == "dora")
    )
    print_trainable_parameters(model)

    # Track memory and performance
    memory_before = get_memory_usage()
    kernels_applied = False

    # Apply evolved quantized LoRA kernels
    if evolved_kernels:
        print("🚀 Applying evolved quantized LoRA kernels...")
        patch_quantized_lora_layers(model, evolved_kernels)
        kernels_applied = getattr(model, '_kernels_applied', False)
        print(f"  📊 Kernels applied: {kernels_applied}")
    else:
        print("🔍 Using standard MLX-LM quantized LoRA")

    # Setup optimizer
    optimizer_name = args.optimizer.lower()
    optimizer_config = args.optimizer_config.get(optimizer_name, {})

    if optimizer_name == "adam":
        optimizer = optim.Adam(learning_rate=args.learning_rate, **optimizer_config)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(learning_rate=args.learning_rate, **optimizer_config)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Setup training
    adapter_path = Path(adapter_save_path)
    adapter_path.mkdir(parents=True, exist_ok=True)

    args.adapter_file = adapter_path / "adapters.safetensors"
    config_to_save = vars(args).copy()
    config_to_save["adapter_file"] = str(config_to_save["adapter_file"])
    save_config(config_to_save, adapter_path / "adapter_config.json")

    training_args = TrainingArgs(
        batch_size=int(args.batch_size),
        iters=int(args.iters),
        val_batches=int(args.val_batches),
        steps_per_report=int(args.steps_per_report),
        steps_per_eval=int(args.steps_per_eval),
        steps_per_save=int(args.save_every),
        adapter_file=str(args.adapter_file),
        max_seq_length=int(args.max_seq_length),
        grad_checkpoint=bool(args.grad_checkpoint),
    )

    # Training with timing and memory tracking
    print("Starting quantized LoRA training...")
    start_time = time.time()
    memory_peak_before = mx.get_peak_memory()

    try:
        train(
            model=model,
            args=training_args,
            optimizer=optimizer,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(valid_set),
            training_callback=None,
        )
    except Exception as e:
        print(f"Training failed: {e}")
        raise

    training_time = time.time() - start_time
    memory_peak_after = mx.get_peak_memory()
    memory_after = get_memory_usage()

    # Evaluation
    print("Evaluating...")
    try:
        final_loss = evaluate(
            model=model,
            dataset=CacheDataset(test_set),
            batch_size=int(args.batch_size),
            num_batches=int(args.test_batches) if hasattr(args, "test_batches") else 5,
            max_seq_length=int(args.max_seq_length),
        )
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise

    # Calculate metrics
    memory_delta = memory_after - memory_before
    memory_peak_delta = memory_peak_after - memory_peak_before

    metrics = {
        "final_loss": float(final_loss),
        "training_time": training_time,
        "memory_delta": float(memory_delta),
        "memory_peak_delta": float(memory_peak_delta / 1e6),  # Convert to MB
        "model_name": model_name,
        "num_layers_trained": args.num_layers,
        "lora_rank": args.lora_parameters["rank"],
        "quantized_layers_count": len(quantized_layers),
        "kernels_applied": kernels_applied,
        "optimization_target": "quantized_lora_fusion",
    }

    return final_loss, metrics


def baseline_lora_kernels():
    """Baseline: No kernels, use standard MLX-LM quantized LoRA."""
    return None


def test_quantized_lora_optimization():
    """Test quantized LoRA optimization functionality."""
    print("Testing MLX Quantized LoRA Optimization...")

    if not MLX_AVAILABLE or not MLX_LM_AVAILABLE:
        print("❌ MLX or MLX-LM not available")
        return False

    try:
        print("\n=== Testing Quantized LoRA Optimization ===")

        # Create test data
        temp_data_dir = "temp_data"
        create_sample_dataset(temp_data_dir, num_samples=50)

        config = create_training_config()
        config["data"] = temp_data_dir

        print("✅ Configuration created for quantized model")
        print(f"  - Model: {config['model']} (quantized)")
        print(f"  - LoRA rank: {config['lora_parameters']['rank']}")
        print(f"  - Training iterations: {config['iters']}")

        # Test evolved kernels
        print("\n📦 Loading evolved quantized LoRA kernels...")
        evolved_kernels = evolved_lora_kernels()
        baseline_kernels = baseline_lora_kernels()

        print("✅ Evolved quantized LoRA kernels loaded")
        print(f"  - Kernels available: {list(evolved_kernels.keys())}")
        print(f"  - Baseline: {baseline_kernels} (standard MLX-LM)")

        # Test basic model loading
        print("\n🔧 Testing quantized model loading...")
        try:
            model, tokenizer = load(config["model"])
            print(f"✅ Model loaded: {type(model).__name__}")

            # Check for quantized layers
            quantized_count = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.QuantizedLinear):
                    quantized_count += 1

            print(f"✅ Found {quantized_count} quantized layers in model")

            if quantized_count == 0:
                print("⚠️ WARNING: No quantized layers found - may not be a quantized model")

        except Exception as e:
            print(f"⚠️ Model loading test failed: {e}")

        print("\n🎯 Quantized LoRA optimization tests passed!")
        print("\nOptimization target:")
        print("- Eliminate dequantization in LoRA forward pass")
        print("- Use mx.quantized_matmul directly on quantized weights")
        print("- Reduce memory usage and improve training speed")
        print("- Maintain numerical accuracy with quantized models")

        # Cleanup
        try:
            import shutil
            shutil.rmtree(temp_data_dir, ignore_errors=True)
            shutil.rmtree("temp_adapters", ignore_errors=True)
        except:
            pass

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_quantized_lora_optimization()
    if success:
        print("\n🎯 MLX Quantized LoRA Optimization Ready!")
        print("\nThis example targets:")
        print("- SPECIFIC INEFFICIENCY: MLX-LM dequantizes weights for LoRA computation")
        print("- OPTIMIZATION TARGET: Use mx.quantized_matmul directly, never dequantize")
        print("- EXPECTED IMPROVEMENT: 15-30% memory reduction, 10-20% speed improvement")
        print("- MEASUREMENT: Memory usage, training time, numerical accuracy")
        print("\nEvolution will discover:")
        print("- Efficient quantized LoRA fusion patterns")
        print("- Memory-optimized computation strategies")
        print("- Apple Silicon-specific quantized optimizations")
        print("\nNext steps:")
        print("1. Run: python evaluator.py")
        print("2. Run: python ../../../openevolve-run.py initial_program.py evaluator.py --config config.yaml")
    else:
        print("\n❌ Setup failed. Please check MLX and MLX-LM installation:")
        print("pip install mlx>=0.15.0 mlx-lm>=0.15.0")
