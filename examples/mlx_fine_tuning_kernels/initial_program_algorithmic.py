"""
Algorithmic MLX LoRA Optimization - OpenEvolve Example

This version provides a NAIVE baseline implementation with clear optimization opportunities
for genuine algorithmic improvements targeting matrix computation strategies.
"""

import math
import time
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import types
import tempfile
import json

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
    print("✅ MLX-LM available for real LoRA fine-tuning")
except ImportError as e:
    print(f"⚠️ MLX-LM not available: {e}")
    MLX_LM_AVAILABLE = False


def create_training_config():
    """Create training configuration for LoRA fine-tuning."""
    return {
        "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "train": True,
        "fine_tune_type": "lora",
        "optimizer": "adam",
        "optimizer_config": {"adam": {}},
        "data": "temp_data",
        "seed": 42,
        "num_layers": 4,
        "batch_size": 2,
        "iters": 10,  # Reduced for faster testing
        "val_batches": 5,
        "learning_rate": 1e-4,
        "steps_per_report": 5,
        "steps_per_eval": 100,
        "adapter_path": "temp_adapters",
        "save_every": 100,
        "max_seq_length": 512,
        "lora_parameters": {"rank": 16, "dropout": 0.0, "scale": 16.0},
        "mask_prompt": False,
        "test": True,
        "test_batches": 10,
        "resume_adapter_file": None,
        "config": None,
        "grad_checkpoint": False,
        "lr_schedule": None,
        "wandb": None,
    }


def create_sample_dataset(output_dir: str, num_samples: int = 20):
    """Create a small sample dataset for LoRA fine-tuning testing."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Simple instruction-following examples
    examples = [
        {"text": "What is the capital of France?\nThe capital of France is Paris."},
        {"text": "Explain machine learning.\nMachine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."},
        {"text": "How do you make tea?\nTo make tea, boil water, add tea leaves or a tea bag to a cup, pour the hot water over the tea, let it steep for 3-5 minutes, then remove the tea leaves or bag."},
        {"text": "What is photosynthesis?\nPhotosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar."},
        {"text": "Name three colors.\nThree colors are red, blue, and green."},
    ]

    # Expand examples to requested number
    expanded_examples = []
    for i in range(num_samples):
        example = examples[i % len(examples)]
        expanded_examples.append(example)

    # Create train, valid, test splits
    train_data = expanded_examples[: int(0.7 * num_samples)]
    valid_data = expanded_examples[int(0.7 * num_samples) : int(0.9 * num_samples)]
    test_data = expanded_examples[int(0.9 * num_samples) :]

    # Ensure at least one example in each split
    if not valid_data:
        valid_data = [train_data[0]]
    if not test_data:
        test_data = [train_data[0]]

    # Write datasets
    for split, data in [("train", train_data), ("valid", valid_data), ("test", test_data)]:
        with open(f"{output_dir}/{split}.jsonl", "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")

    print(f"✅ Created dataset with {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test examples")


def evolved_lora_kernels():
    """
    LoRA kernel implementations with NAIVE baseline for algorithmic optimization.
    
    EVOLUTION TARGET: The optimized_lora_matmul function below contains a deliberately
    NAIVE implementation with clear optimization opportunities for genuine improvements.
    """

    if not MLX_LM_AVAILABLE:
        raise ImportError("MLX-LM is required for LoRA kernel optimization")

    # Helper functions available for optimization (can be used by evolved implementations)
    def compute_optimal_chunk_size(tensor_shape: Tuple[int, ...], max_memory_mb: int = 512) -> int:
        """Compute optimal chunk size based on tensor shape and memory constraints."""
        batch_size, seq_len, features = tensor_shape
        # Estimate memory per element (float32 = 4 bytes)
        memory_per_token = batch_size * features * 4 / (1024 * 1024)  # MB per token
        max_tokens = max_memory_mb / memory_per_token
        return max(32, min(seq_len, int(max_tokens)))

    def estimate_computation_cost(x_shape: Tuple[int, ...], rank: int, strategy: str) -> float:
        """Estimate computational cost for different strategies."""
        batch_size, seq_len, input_features = x_shape
        
        if strategy == "standard":
            # Cost: (batch * seq * input * rank) + (batch * seq * rank * output)
            return batch_size * seq_len * (input_features * rank + rank * input_features)
        elif strategy == "precompute":
            # Cost: (input * rank * output) + (batch * seq * input * output)
            return input_features * rank * input_features + batch_size * seq_len * input_features * input_features
        else:
            return float('inf')

    def get_memory_info() -> Dict[str, float]:
        """Get current memory usage information."""
        try:
            import psutil
            process = psutil.Process()
            return {
                "used_mb": process.memory_info().rss / 1024 / 1024,
                "available_mb": psutil.virtual_memory().available / 1024 / 1024
            }
        except:
            return {"used_mb": 0, "available_mb": 1024}

    # EVOLVE-BLOCK-START
    @mx.compile
    def optimized_lora_matmul(x, lora_a, lora_b, scale):
        """
        NAIVE LoRA matrix multiplication - INTENTIONALLY SUBOPTIMAL for evolution target.
        
        This implementation always uses the same strategy regardless of tensor characteristics,
        creating clear optimization opportunities:
        
        CURRENT ISSUES:
        1. Always uses (x @ A) @ B order - may be inefficient for small ranks
        2. No chunking for large sequences - can cause memory issues  
        3. No consideration of tensor shapes for optimization
        4. No fusion opportunities - could use mx.fused() for efficiency
        5. Processes entire sequence at once regardless of size
        6. Inefficient scaling order - could be optimized
        
        INPUT CHARACTERISTICS:
        - x: (batch_size, seq_len, input_features) 
        - lora_a: (input_features, rank)
        - lora_b: (rank, output_features)
        - rank varies: 8, 16, 32, 64, 128
        - seq_len varies: 64, 128, 256, 512, 1024
        
        OPTIMIZATION OPPORTUNITIES:
        1. For small rank: consider pre-computing lora_a @ lora_b
        2. For large sequences: implement chunking
        3. For memory efficiency: use blocking strategies
        4. For MLX: leverage fused operations with mx.fused()
        5. Optimize scaling operation placement
        6. Use adaptive algorithms based on tensor shapes
        """
        # NAIVE IMPLEMENTATION - Always same strategy, no optimization
        # This is deliberately inefficient and should be improved by evolution
        
        # Always use the same order regardless of shapes (suboptimal)
        # Could be optimized: for small ranks, (A @ B) first might be better
        temp = mx.matmul(x, lora_a)
        result = mx.matmul(temp, lora_b)
        
        # Always apply scale at the end (could be optimized)
        # Could be optimized: scale could be applied earlier or fused
        scaled_result = scale * result
        
        # No chunking for large sequences (memory inefficient)
        # No fusion optimizations (performance suboptimal)
        # No adaptive algorithm selection (one-size-fits-all approach)
        
        return scaled_result
    # EVOLVE-BLOCK-END

    # All other kernel functions remain unchanged for stability
    class OptimizedLoRALinear(nn.Module):
        """Simplified LoRA linear layer that uses the optimized matmul function."""

        def __init__(self, original_lora_layer, r=16, alpha=16, dropout=0.0, scale=None):
            super().__init__()
            self.base_layer = getattr(original_lora_layer, 'linear', original_lora_layer)
            self.r = r
            self.alpha = alpha
            self.dropout = dropout
            self.scale = scale if scale is not None else alpha / r

            # Initialize LoRA weights
            if hasattr(self.base_layer, 'weight'):
                in_features = self.base_layer.weight.shape[1]
                out_features = self.base_layer.weight.shape[0]
            else:
                in_features = getattr(original_lora_layer, 'in_features', 512)
                out_features = getattr(original_lora_layer, 'out_features', 512)

            self.lora_a = mx.random.normal((in_features, r)) * 0.01
            self.lora_b = mx.zeros((r, out_features))

        def __call__(self, x):
            """Forward pass using the optimized matmul function."""
            base_out = self.base_layer(x)
            # Use the optimized matmul function (this is where evolution happens)
            lora_out = optimized_lora_matmul(x, self.lora_a, self.lora_b, self.scale)
            return base_out + lora_out

    # Standard utility functions (unchanged)
    def optimized_lora_forward_pass(model, x, use_kernels=True):
        """Standard forward pass (unchanged)."""
        if not use_kernels:
            return model(x)
        try:
            return model(x)
        except Exception:
            return model._original_forward(x) if hasattr(model, "_original_forward") else model(x)

    def optimized_gradient_computation(loss, model, use_kernels=True):
        """Standard gradient computation (unchanged)."""
        if not use_kernels:
            def loss_fn(m):
                return loss
            return mx.value_and_grad(loss_fn)(model)[1]
        
        try:
            def loss_fn(m):
                return loss
            @mx.compile
            def compiled_grad_fn(model_params):
                return mx.grad(loss_fn)(model_params)
            return compiled_grad_fn(model)
        except Exception:
            def loss_fn(m):
                return loss
            return mx.value_and_grad(loss_fn)(model)[1]

    @mx.compile
    def optimized_parameter_update(params, grads, lr):
        """Standard parameter update (unchanged)."""
        updated_params = {}
        for key in params:
            if key in grads:
                updated_params[key] = params[key] - lr * grads[key]
            else:
                updated_params[key] = params[key]
        return updated_params

    def memory_efficient_loss_computation(logits, targets, chunk_size=1024):
        """Standard loss computation (unchanged)."""
        if logits.shape[-1] <= chunk_size:
            return nn.losses.cross_entropy(logits, targets, reduction="mean")

        batch_size, seq_len, vocab_size = logits.shape
        total_loss = 0.0
        num_chunks = (vocab_size + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, vocab_size)

            logits_chunk = logits[:, :, start_idx:end_idx]
            targets_chunk = mx.where(
                (targets >= start_idx) & (targets < end_idx),
                targets - start_idx,
                -1,
            )

            valid_mask = targets_chunk >= 0
            if mx.any(valid_mask):
                chunk_loss = nn.losses.cross_entropy(logits_chunk, targets_chunk, reduction="mean")
                total_loss += chunk_loss * mx.mean(valid_mask.astype(mx.float32))

        return total_loss / num_chunks

    return {
        "optimized_lora_linear_class": OptimizedLoRALinear,
        "optimized_lora_matmul": optimized_lora_matmul,  # This is the evolution target
        "optimized_lora_forward_pass": optimized_lora_forward_pass,
        "optimized_gradient_computation": optimized_gradient_computation,
        "optimized_parameter_update": optimized_parameter_update,
        "memory_efficient_loss_computation": memory_efficient_loss_computation,
        # Helper functions available for optimization
        "compute_optimal_chunk_size": compute_optimal_chunk_size,
        "estimate_computation_cost": estimate_computation_cost,
        "get_memory_info": get_memory_info,
    }


def patch_model_with_kernels(model, evolved_kernels):
    """Simplified model patching focusing on LoRA layer replacement."""
    if not evolved_kernels:
        print("  🔍 No evolved kernels to apply - using standard MLX-LM")
        model._kernels_applied = False
        return

    print(f"🚀 Patching model with evolved kernels...")

    try:
        if not hasattr(model, "_original_forward"):
            model._original_forward = model.__call__

        OptimizedLoRALinear = evolved_kernels.get("optimized_lora_linear_class")
        replaced_count = 0
        
        if OptimizedLoRALinear:
            print("  🔧 Replacing LoRA layers with optimized versions...")
            
            all_modules = list(model.named_modules())
            lora_layers_to_replace = []
            
            # Find LoRA layers
            for name, module in all_modules:
                module_type = type(module).__name__
                has_lora = (
                    (hasattr(module, 'lora_a') and hasattr(module, 'lora_b')) or
                    (hasattr(module, 'A') and hasattr(module, 'B')) or
                    any('lora' in attr.lower() for attr in dir(module) if not attr.startswith('_')) or
                    'lora' in module_type.lower()
                )
                
                param_names = []
                try:
                    param_names = list(dict(module.named_parameters()).keys())
                except:
                    pass
                
                has_lora_params = any('lora' in p.lower() for p in param_names)
                
                if has_lora or has_lora_params:
                    lora_layers_to_replace.append((name, module))
            
            # Replace LoRA layers
            for layer_name, lora_layer in lora_layers_to_replace:
                try:
                    # Extract LoRA matrices
                    lora_a = None
                    lora_b = None
                    
                    if hasattr(lora_layer, 'lora_a') and hasattr(lora_layer, 'lora_b'):
                        lora_a, lora_b = lora_layer.lora_a, lora_layer.lora_b
                    elif hasattr(lora_layer, 'A') and hasattr(lora_layer, 'B'):
                        lora_a, lora_b = lora_layer.A, lora_layer.B
                    else:
                        # Try parameters
                        try:
                            params = dict(lora_layer.named_parameters())
                            param_names = list(params.keys())
                            a_candidates = [p for p in param_names if 'a' in p.lower() or 'down' in p.lower()]
                            b_candidates = [p for p in param_names if 'b' in p.lower() or 'up' in p.lower()]
                            
                            if a_candidates and b_candidates:
                                lora_a = params[a_candidates[0]]
                                lora_b = params[b_candidates[0]]
                        except Exception:
                            pass
                    
                    if lora_a is None or lora_b is None:
                        continue
                    
                    # Create optimized version
                    # Determine rank from lora_a shape: (input_features, rank) or (rank, input_features)
                    if lora_a.shape[0] < lora_a.shape[1]:
                        r = lora_a.shape[0]  # (rank, input_features)
                    else:
                        r = lora_a.shape[1]  # (input_features, rank)
                    optimized_layer = OptimizedLoRALinear(
                        original_lora_layer=lora_layer,
                        r=r,
                        alpha=getattr(lora_layer, 'alpha', 16),
                        dropout=getattr(lora_layer, 'dropout', 0.0),
                        scale=getattr(lora_layer, 'scale', None)
                    )
                    
                    # Copy weights
                    optimized_layer.lora_a = lora_a
                    optimized_layer.lora_b = lora_b
                    
                    # Replace in model (simplified navigation)
                    name_parts = layer_name.split('.')
                    if len(name_parts) == 1:
                        setattr(model, name_parts[0], optimized_layer)
                    else:
                        parent = model
                        for part in name_parts[:-1]:
                            if hasattr(parent, part):
                                parent = getattr(parent, part)
                            elif part.isdigit() and hasattr(parent, '__getitem__'):
                                parent = parent[int(part)]
                        
                        final_attr = name_parts[-1]
                        if hasattr(parent, final_attr):
                            setattr(parent, final_attr, optimized_layer)
                        elif final_attr.isdigit() and hasattr(parent, '__setitem__'):
                            parent[int(final_attr)] = optimized_layer
                    
                    replaced_count += 1
                    
                except Exception as e:
                    print(f"    ⚠️ Failed to replace {layer_name}: {e}")

        model._evolved_kernels = evolved_kernels
        model._has_evolved_kernels = True
        model._kernels_applied = replaced_count > 0

        print(f"  ✅ Replaced {replaced_count} LoRA layers")
        print(f"  📊 Kernels applied: {getattr(model, '_kernels_applied', False)}")

    except Exception as e:
        print(f"❌ ERROR during patching: {e}")
        model._kernels_applied = False


def unpatch_model(model):
    """Remove evolved kernel patches from model."""
    if hasattr(model, "_kernels_applied") and not getattr(model, "_kernels_applied", True):
        return

    try:
        if hasattr(model, "_original_forward"):
            original_forward = getattr(model, "_original_forward", None)
            if original_forward:
                model.__call__ = original_forward
    except Exception:
        pass

    attributes_to_clean = ["_original_forward", "_evolved_kernels", "_has_evolved_kernels", "_kernels_applied"]
    for attr_name in attributes_to_clean:
        if hasattr(model, attr_name):
            try:
                delattr(model, attr_name)
            except (AttributeError, TypeError):
                try:
                    setattr(model, attr_name, None)
                except Exception:
                    pass


def standard_lora_fine_tuning_with_kernels(
    model_name: str,
    train_data_path: str,
    config: Dict[str, Any],
    adapter_save_path: str = "temp_adapters",
    evolved_kernels: Optional[Dict] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Standard MLX-LM LoRA fine-tuning with optional evolved kernel optimizations."""
    
    # Set random seed for reproducibility
    mx.random.seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))

    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    model, tokenizer = load(model_name)

    # Convert config to namespace
    args = types.SimpleNamespace(**config)
    args.data = train_data_path

    # Load datasets
    print("Loading datasets...")
    train_set, valid_set, test_set = load_dataset(args, tokenizer)

    # Apply LoRA
    print("Applying LoRA...")
    model.freeze()
    linear_to_lora_layers(
        model, args.num_layers, args.lora_parameters, use_dora=(args.fine_tune_type == "dora")
    )
    print_trainable_parameters(model)

    # Apply evolved kernels
    kernels_actually_applied = False
    if evolved_kernels:
        print("🚀 Applying evolved kernels...")
        patch_model_with_kernels(model, evolved_kernels)
        kernels_actually_applied = getattr(model, '_kernels_applied', False)
        print(f"  📊 Kernels applied: {kernels_actually_applied}")
    else:
        print("🔍 Using standard MLX-LM")

    # Setup optimizer
    optimizer_name = args.optimizer.lower()
    optimizer_config = args.optimizer_config.get(optimizer_name, {})

    if optimizer_name == "adam":
        optimizer = optim.Adam(learning_rate=args.learning_rate, **optimizer_config)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(learning_rate=args.learning_rate, **optimizer_config)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Create adapter save directory
    adapter_path = Path(adapter_save_path)
    adapter_path.mkdir(parents=True, exist_ok=True)

    # Save configuration
    args.adapter_file = adapter_path / "adapters.safetensors"
    config_to_save = vars(args).copy()
    config_to_save["adapter_file"] = str(config_to_save["adapter_file"])
    save_config(config_to_save, adapter_path / "adapter_config.json")

    # Training arguments
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

    # Training
    print("Starting training...")
    start_time = time.time()

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
    finally:
        if evolved_kernels:
            unpatch_model(model)

    training_time = time.time() - start_time

    # Evaluation
    print("Evaluating...")
    try:
        final_loss = evaluate(
            model=model,
            dataset=CacheDataset(test_set),
            batch_size=int(args.batch_size),
            num_batches=int(args.test_batches) if hasattr(args, "test_batches") else 10,
            max_seq_length=int(args.max_seq_length),
        )
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise

    metrics = {
        "final_loss": float(final_loss),
        "training_time": training_time,
        "model_name": model_name,
        "num_layers_trained": args.num_layers,
        "lora_rank": args.lora_parameters["rank"],
        "used_evolved_kernels": kernels_actually_applied,
        "kernels_used": kernels_actually_applied,
        "kernels_provided": evolved_kernels is not None,
        "kernels_applied": kernels_actually_applied,
    }

    return final_loss, metrics


def baseline_lora_kernels():
    """Baseline: Return None to use standard MLX-LM without any optimizations."""
    return None


def test_lora_functionality():
    """Test basic LoRA functionality."""
    print("Testing Algorithmic MLX-LM LoRA Optimization...")

    if not MLX_AVAILABLE or not MLX_LM_AVAILABLE:
        print("❌ MLX or MLX-LM not available")
        return False

    try:
        # Create test data
        temp_data_dir = "temp_data"
        create_sample_dataset(temp_data_dir, num_samples=20)

        # Test configuration
        config = create_training_config()
        config["data"] = temp_data_dir

        print("✅ Configuration created")
        print(f"  - Model: {config['model']}")
        print(f"  - LoRA rank: {config['lora_parameters']['rank']}")

        # Test kernels
        print("\n📦 Testing evolved kernels...")
        evolved_kernels = evolved_lora_kernels()
        baseline_kernels = baseline_lora_kernels()

        print("✅ Kernels loaded successfully")
        print(f"  - Evolved kernels: {list(evolved_kernels.keys())}")
        print(f"  - Evolution target: optimized_lora_matmul (NAIVE baseline)")
        print(f"  - Helper functions: compute_optimal_chunk_size, estimate_computation_cost, get_memory_info")

        # Test the naive implementation for obvious inefficiencies
        print("\n🔍 Analyzing naive baseline implementation...")
        kernel_func = evolved_kernels["optimized_lora_matmul"]
        print("  ⚠️ Current implementation issues:")
        print("    - Always clears cache (inefficient)")  
        print("    - Always uses same matmul order (not adaptive)")
        print("    - Always forces evaluation (unnecessary)")
        print("    - No chunking for large sequences")
        print("    - No shape-based optimization")
        print("  🎯 Optimization opportunities identified!")

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
    success = test_lora_functionality()
    if success:
        print("\n🎯 Algorithmic MLX LoRA Optimization Ready!")
        print("\nEVOLUTION TARGET: optimized_lora_matmul function")
        print("- Current: NAIVE implementation with clear inefficiencies")
        print("- Goal: 15%+ algorithmic improvement via adaptive strategies")
        print("- Opportunities: Matrix order optimization, chunking, memory management")
        print("\nNaive baseline provides genuine optimization opportunities:")
        print("1. 🔧 Adaptive matrix multiplication order based on rank")
        print("2. 🔧 Sequence chunking for memory efficiency") 
        print("3. 🔧 Shape-based algorithm selection")
        print("4. 🔧 MLX-specific fusion optimizations")
        print("5. 🔧 Conditional memory management")
        print("\nNext steps:")
        print("1. Run: python evaluator.py")
        print("2. Run: python ../../../openevolve-run.py initial_program_algorithmic.py evaluator.py --config config_algorithmic.yaml")
    else:
        print("\n❌ Setup failed. Please check MLX and MLX-LM installation")
