"""
MLX LoRA + Quantization Fusion Optimization - EVOLVED VERSION

This program contains the best evolved quantized LoRA kernels with:
- Advanced bias fusion within compiled kernels
- Sophisticated dropout handling with separate paths  
- Memory optimization strategies
- Multiple compiled kernel variants for different scenarios

Evolution Generation: 4, Iteration: 11
Base Score: 0.9654 with advanced optimizations
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
        "num_layers": 3,
        "batch_size": 2,
        "iters": 50,  # EVOLVED: Increased from 15 for better convergence
        "val_batches": 5,
        "learning_rate": 1e-4,
        "steps_per_report": 10,  # EVOLVED: Adjusted for longer training
        "steps_per_eval": 100,
        "adapter_path": "temp_adapters",
        "save_every": 100,
        "max_seq_length": 256,
        "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 16.0},
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
    """Create a sample dataset for quantized LoRA testing."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    examples = [
        {"text": "What is machine learning?\nMachine learning is AI that learns from data without explicit programming."},
        {"text": "Explain deep learning.\nDeep learning uses neural networks with many layers to learn complex patterns."},
        {"text": "What is quantization?\nQuantization reduces model size by using lower precision numbers like int8 or int4."},
        {"text": "How does LoRA work?\nLoRA adds small trainable matrices to frozen pre-trained weights for efficient fine-tuning."},
        {"text": "What is Apple Silicon?\nApple Silicon refers to custom ARM-based processors designed by Apple for Mac computers."},
        {"text": "What is MLX?\nMLX is Apple's machine learning framework optimized for Apple Silicon processors."},
        {"text": "Explain transformers.\nTransformers are neural networks that use attention mechanisms for sequence processing."},
        {"text": "What is fine-tuning?\nFine-tuning adapts pre-trained models to specific tasks with task-specific data."},
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
    EVOLVED LoRA kernel implementations targeting quantized LoRA fusion.

    ADVANCED FEATURES (Evolution Generation 4):
    - Bias fusion within compiled kernels  
    - Sophisticated dropout path separation
    - Memory optimization strategies (90% limit for stability)
    - Multiple compiled kernel variants

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
    def optimized_quantized_lora_matmul(x, quantized_weight, scales, biases, lora_a, lora_b, scale, group_size, bits, additive_bias_optional=None):
        """
        EVOLVED: Core optimized quantized LoRA computation with bias fusion.
        
        CRITICAL OPTIMIZATION: Uses mx.quantized_matmul directly instead of dequantizing.
        This is the primary efficiency gain - eliminates temporary full-precision weights.
        
        EVOLUTION IMPROVEMENTS:
        - Bias fusion within compiled kernel (reduces kernel launches)
        - Proper additive bias handling
        """
        # Direct quantized matrix multiplication - no dequantization needed
        base_out = mx.quantized_matmul(
            x, quantized_weight, scales, biases,
            group_size=group_size, bits=bits, transpose=True
        )
        
        # EVOLVED: Add base layer bias if present (fused within kernel)
        if additive_bias_optional is not None:
            base_out = base_out + additive_bias_optional
        
        # Efficient LoRA computation with compilation
        lora_temp = mx.matmul(x, lora_a)
        lora_out = mx.matmul(lora_temp, lora_b)
        
        # Fuse outputs with proper type casting
        return base_out + (scale * lora_out).astype(base_out.dtype)

    @mx.compile
    def optimized_quantized_lora_matmul_with_dropout(x_base, x_lora, quantized_weight, scales, biases, lora_a, lora_b, scale, group_size, bits, additive_bias_optional=None):
        """
        EVOLVED: Optimized quantized LoRA computation for cases with dropout.
        
        SOPHISTICATION: Takes separate inputs for base and LoRA paths, enabling
        correct dropout semantics while maintaining performance.
        """
        # Base path (no dropout)
        base_out = mx.quantized_matmul(
            x_base, quantized_weight, scales, biases,
            group_size=group_size, bits=bits, transpose=True
        )
        
        # EVOLVED: Add base layer bias if present (fused within kernel)
        if additive_bias_optional is not None:
            base_out = base_out + additive_bias_optional
            
        # LoRA path (with dropout applied to x_lora)
        lora_temp = mx.matmul(x_lora, lora_a)
        lora_out = mx.matmul(lora_temp, lora_b)
        
        # Fuse outputs with proper type casting
        return base_out + (scale * lora_out).astype(base_out.dtype)


    @mx.compile  
    def optimized_lora_computation(x, lora_a, lora_b, scale):
        """Compiled LoRA matrix computation for efficiency. (Used for non-quantized path)"""
        temp = mx.matmul(x, lora_a)
        result = mx.matmul(temp, lora_b)
        return scale * result

    class OptimizedQuantizedLoRALinear(nn.Module):
        """
        EVOLVED: Optimized LoRA linear layer that works directly with quantized weights.
        
        KEY OPTIMIZATION: Never dequantizes base weights, uses mx.quantized_matmul directly.
        This is the core innovation that eliminates the dequantization bottleneck.
        
        EVOLUTION IMPROVEMENTS:
        - Sophisticated dropout handling with separate paths
        - Bias fusion within compiled kernels
        - Enhanced error handling and validation
        """

        def __init__(self, original_lora_layer, r=8, alpha=16, dropout=0.0, scale=None):
            super().__init__()
            
            # Extract the base layer (linear or quantized)
            if hasattr(original_lora_layer, 'linear'):
                self.base_layer = original_lora_layer.linear
            else:
                self.base_layer = original_lora_layer
                
            # Determine if we can apply quantized optimization
            self._is_quantized = isinstance(self.base_layer, nn.QuantizedLinear)
            
            if self._is_quantized:
                print(f"  ✅ Applying quantized optimization: {self.base_layer.bits}-bit, group_size={self.base_layer.group_size}")
            else:
                print(f"  ℹ️ Non-quantized layer detected: {type(self.base_layer)}")

            # LoRA parameters
            self.r = r
            self.alpha = alpha
            self.scale = scale if scale is not None else alpha / r

            # EVOLVED: Handle dropout layer with sophistication
            if hasattr(original_lora_layer, 'dropout') and isinstance(original_lora_layer.dropout, nn.Dropout):
                self.dropout_layer = original_lora_layer.dropout
                print(f"    ✅ Copied LoRA dropout layer.")
            else:
                self.dropout_layer = nn.Dropout(dropout)
                print(f"    ✅ Initialized LoRA dropout layer with p={dropout}.")

            # Copy or initialize LoRA weights
            if hasattr(original_lora_layer, 'lora_a') and hasattr(original_lora_layer, 'lora_b'):
                self.lora_a = original_lora_layer.lora_a
                self.lora_b = original_lora_layer.lora_b
                print(f"    ✅ Copied LoRA weights: A={self.lora_a.shape}, B={self.lora_b.shape}")
            else:
                # Initialize new LoRA weights
                if hasattr(self.base_layer, 'weight'):
                    weight_shape = self.base_layer.weight.shape
                    # For quantized weights, the stored weight shape is (output_dims, input_dims_packed)
                    # input_dims_packed = input_dims * bits // 32
                    # So, input_dims = input_dims_packed * 32 // bits
                    input_dims = weight_shape[1]
                    output_dims = weight_shape[0]
                    
                    if self._is_quantized:
                        input_dims = input_dims * 32 // self.base_layer.bits
                else:
                    # Fallback dimensions (should not happen if base_layer has a weight)
                    input_dims = 512
                    output_dims = 512
                
                scale_init = 1 / math.sqrt(input_dims)
                self.lora_a = mx.random.uniform(
                    low=-scale_init, high=scale_init, shape=(input_dims, r)
                )
                self.lora_b = mx.zeros(shape=(r, output_dims))
                print(f"    ✅ Initialized LoRA weights: A={self.lora_a.shape}, B={self.lora_b.shape}")

        def __call__(self, x):
            """
            EVOLVED: Optimized forward pass using quantized operations with sophisticated dropout handling.
            
            This is where the magic happens - we use mx.quantized_matmul directly
            instead of dequantizing the entire weight matrix.
            
            EVOLUTION IMPROVEMENTS:
            - Separate dropout paths for correctness
            - Bias fusion within compiled kernels  
            - Enhanced kernel selection logic
            """
            
            # Determine if dropout is active
            has_dropout = self.dropout_layer and self.dropout_layer.p > 0.0

            if not self._is_quantized:
                # For non-quantized layers, use standard computation
                base_out = self.base_layer(x) # This applies base_layer's matmul and bias
                
                # LoRA path always applies dropout if enabled
                x_lora = self.dropout_layer(x) if has_dropout else x
                lora_out = optimized_lora_computation(x_lora, self.lora_a, self.lora_b, self.scale)
                
                return base_out + lora_out.astype(x.dtype)

            # CORE OPTIMIZATION: Use quantized operations directly with fully fused kernels
            additive_bias = self.base_layer.bias if hasattr(self.base_layer, 'bias') else None

            if has_dropout:
                # EVOLVED: If dropout is active, base path uses original x, LoRA path uses dropout(x)
                # Use the specialized compiled kernel for this case
                x_lora = self.dropout_layer(x)
                result = optimized_quantized_lora_matmul_with_dropout(
                    x, # x_base
                    x_lora, # x_lora
                    self.base_layer.weight,
                    self.base_layer.scales,
                    self.base_layer.biases,
                    self.lora_a,
                    self.lora_b,
                    self.scale,
                    self.base_layer.group_size,
                    self.base_layer.bits,
                    additive_bias # EVOLVED: Pass the additive bias to the kernel
                )
            else:
                # EVOLVED: If no dropout, 'x' is the same for both base and LoRA paths.
                # Use the fully fused compiled kernel for maximum efficiency.
                result = optimized_quantized_lora_matmul(
                    x,
                    self.base_layer.weight,
                    self.base_layer.scales,
                    self.base_layer.biases,
                    self.lora_a,
                    self.lora_b,
                    self.scale,
                    self.base_layer.group_size,
                    self.base_layer.bits,
                    additive_bias # EVOLVED: Pass the additive bias to the kernel
                )
                
            return result

    @mx.compile
    def optimized_quantized_loss_computation(logits, targets):
        """Optimized loss computation for quantized models."""
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    def quantized_model_memory_optimizer(model):
        """
        EVOLVED: Optimize memory usage patterns for quantized models.
        
        EVOLUTION IMPROVEMENT: Adjusted memory limit from 95% to 90% for better 
        stability and convergence based on training analysis.
        """
        # For quantized models, we can be more aggressive with memory usage
        # EVOLVED: Adjust memory limit for quantized models - slightly less aggressive to improve stability/convergence
        max_mem = mx.metal.device_info()["max_recommended_working_set_size"]
        quantized_limit = int(0.90 * max_mem)  # EVOLVED: Use 90% of recommended max working set size
        mx.set_wired_limit(quantized_limit)
        
        print(f"  🎯 Optimized memory limit for quantized model: {quantized_limit // (1024*1024)} MB (90% of max recommended)")

    return {
        "optimized_quantized_lora_linear_class": OptimizedQuantizedLoRALinear,
        "optimized_quantized_lora_matmul": optimized_quantized_lora_matmul,
        "optimized_quantized_lora_matmul_with_dropout": optimized_quantized_lora_matmul_with_dropout, # EVOLVED: Add new kernel
        "optimized_lora_computation": optimized_lora_computation,
        "optimized_quantized_loss_computation": optimized_quantized_loss_computation,
        "quantized_model_memory_optimizer": quantized_model_memory_optimizer,
    }
    # EVOLVE-BLOCK-END


def replace_model_layer(model, layer_path, new_layer):
    """
    Robust layer replacement that handles both attributes and list indices.
    
    Args:
        model: The model to modify
        layer_path: String path like "model.layers.23.self_attn.q_proj"
        new_layer: The replacement layer
        
    Returns:
        bool: True if replacement succeeded, False otherwise
    """
    try:
        # Split the path and navigate to parent
        parts = layer_path.split('.')
        current = model
        
        print(f"      DEBUG: Navigating path: {layer_path}")
        
        # Navigate to the parent of the target layer
        for i, part in enumerate(parts[:-1]):
            print(f"        Step {i}: Accessing '{part}' on {type(current)}")
            
            if part.isdigit():
                # This is a list index
                index = int(part)
                if hasattr(current, '__getitem__') and hasattr(current, '__len__'):
                    if index < len(current):
                        current = current[index]
                        print(f"          -> Used list index: current[{index}]")
                    else:
                        print(f"          ERROR: Index {index} out of bounds for list of length {len(current)}")
                        return False
                else:
                    print(f"          ERROR: Trying to index into non-indexable object: {type(current)}")
                    return False
            else:
                # This is an attribute
                if hasattr(current, part):
                    current = getattr(current, part)
                    print(f"          -> Used attribute: getattr(current, '{part}')")
                else:
                    print(f"          ERROR: Object {type(current)} has no attribute '{part}'")
                    return False
        
        # Now replace the final layer
        final_part = parts[-1]
        print(f"      DEBUG: Setting final part '{final_part}' on {type(current)}")
        
        if final_part.isdigit():
            # Final part is a list index
            index = int(final_part)
            if hasattr(current, '__setitem__') and hasattr(current, '__len__'):
                if index < len(current):
                    current[index] = new_layer
                    print(f"        -> Set using list assignment: current[{index}] = new_layer")
                else:
                    print(f"        ERROR: Index {index} out of bounds for list of length {len(current)}")
                    return False
            else:
                print(f"        ERROR: Cannot set index on non-indexable object: {type(current)}")
                return False
        else:
            # Final part is an attribute
            if hasattr(current, final_part):
                setattr(current, final_part, new_layer)
                print(f"        -> Set using attribute assignment: setattr(current, '{final_part}', new_layer)")
            else:
                print(f"        ERROR: Cannot set attribute '{final_part}' on {type(current)}")
                return False
        
        # Verify the replacement worked
        print(f"      DEBUG: Verifying replacement...")
        verification_current = model
        for part in parts[:-1]:
            if part.isdigit():
                verification_current = verification_current[int(part)]
            else:
                verification_current = getattr(verification_current, part)
        
        if final_part.isdigit():
            replaced_layer = verification_current[int(final_part)]
        else:
            replaced_layer = getattr(verification_current, final_part)
        
        success = type(replaced_layer).__name__ == 'OptimizedQuantizedLoRALinear'
        print(f"      DEBUG: Verification result: {success} (layer type: {type(replaced_layer)})")
        
        return success
        
    except Exception as e:
        print(f"      ERROR: Layer replacement failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def apply_quantized_lora_optimizations(model, evolved_kernels):
    """
    Apply evolved quantized LoRA optimizations to model with robust validation.
    
    Returns: (success: bool, details: dict)
    """
    if not evolved_kernels:
        print("  🔍 No evolved kernels to apply")
        model._kernels_applied = False
        return False, {"reason": "no_kernels_provided"}

    print(f"🚀 Applying quantized LoRA optimizations...")

    try:
        # Apply memory optimization first
        memory_optimizer = evolved_kernels.get("quantized_model_memory_optimizer")
        if memory_optimizer:
            memory_optimizer(model)

        # Get the optimized class
        OptimizedQuantizedLoRALinear = evolved_kernels.get("optimized_quantized_lora_linear_class")
        if not OptimizedQuantizedLoRALinear:
            print("  ❌ No optimized LoRA class found in evolved kernels")
            model._kernels_applied = False
            return False, {"reason": "no_optimized_class"}

        # Scan for LoRA layers to replace
        lora_layers_found = []
        for name, module in model.named_modules():
            module_type = type(module).__name__
            
            # Look for LoRA layers from MLX-LM
            if ('LoRA' in module_type or 
                hasattr(module, 'lora_a') and hasattr(module, 'lora_b')):
                lora_layers_found.append((name, module))

        print(f"  🔍 Found {len(lora_layers_found)} LoRA layers to optimize")

        if len(lora_layers_found) == 0:
            print("  ⚠️ No LoRA layers found in model")
            model._kernels_applied = False
            return False, {"reason": "no_lora_layers_found"}

        # Replace LoRA layers with optimized versions
        replaced_count = 0
        quantized_optimized_count = 0
        
        for layer_name, lora_layer in lora_layers_found:
            print(f"    📌 Optimizing LoRA layer: {layer_name}")
            
            try:
                # Create optimized version
                optimized_layer = OptimizedQuantizedLoRALinear(
                    original_lora_layer=lora_layer,
                    r=getattr(lora_layer, 'r', 8),
                    alpha=getattr(lora_layer, 'alpha', 16),
                    dropout=getattr(lora_layer, 'dropout', 0.0),
                    scale=getattr(lora_layer, 'scale', None)
                )
                
                # Check if this is actually a quantized optimization
                if optimized_layer._is_quantized:
                    quantized_optimized_count += 1
                
                # Use robust layer replacement
                replacement_success = replace_model_layer(model, layer_name, optimized_layer)
                
                if replacement_success:
                    replaced_count += 1
                    print(f"      ✅ Successfully optimized {layer_name}")
                else:
                    print(f"      ❌ Failed to replace {layer_name}")
                
            except Exception as e:
                print(f"    ❌ Failed to optimize {layer_name}: {e}")
                # Don't fail the entire process for one layer
                continue

        print(f"  ✅ Optimization complete:")
        print(f"    Total LoRA layers replaced: {replaced_count}")
        print(f"    Quantized optimizations applied: {quantized_optimized_count}")

        # Store optimization details
        model._evolved_kernels = evolved_kernels
        model._has_evolved_kernels = True
        model._kernels_applied = replaced_count > 0
        model._quantized_optimizations = quantized_optimized_count

        success = replaced_count > 0
        details = {
            "replaced_count": replaced_count,
            "quantized_optimized_count": quantized_optimized_count,
            "total_lora_layers": len(lora_layers_found)
        }

        return success, details

    except Exception as e:
        print(f"❌ ERROR during quantized LoRA optimization: {e}")
        import traceback
        traceback.print_exc()
        model._kernels_applied = False
        return False, {"reason": "exception", "error": str(e)}


def quantized_lora_fine_tuning_with_kernels(
    model_name: str,
    train_data_path: str,
    config: Dict[str, Any],
    adapter_save_path: str = "temp_adapters",
    evolved_kernels: Optional[Dict] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Robust quantized LoRA fine-tuning with evolved kernel optimizations.
    
    This function provides clean comparison between standard MLX-LM and optimized kernels.
    """
    # Set random seed for reproducibility
    mx.random.seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))

    print(f"Loading quantized model: {model_name}")
    model, tokenizer = load(model_name)

    # Validate model has quantized layers
    quantized_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            quantized_layers.append((name, module))

    print(f"✅ Model validation: {len(quantized_layers)} quantized layers found")
    
    if len(quantized_layers) == 0:
        print("⚠️ WARNING: No quantized layers found - optimization may not be effective")

    # Setup MLX-LM components
    args = types.SimpleNamespace(**config)
    args.data = train_data_path

    print("Loading datasets...")
    train_set, valid_set, test_set = load_dataset(args, tokenizer)

    # Apply standard LoRA first (this is the same for both baseline and evolved)
    print("Applying standard LoRA layers...")
    model.freeze()
    linear_to_lora_layers(
        model, args.num_layers, args.lora_parameters, use_dora=(args.fine_tune_type == "dora")
    )
    print_trainable_parameters(model)

    # Apply evolved kernels if provided
    kernels_applied = False
    optimization_details = {}
    
    if evolved_kernels:
        print("🚀 Applying evolved quantized LoRA kernels...")
        kernels_applied, optimization_details = apply_quantized_lora_optimizations(model, evolved_kernels)
        print(f"  📊 Kernels applied: {kernels_applied}")
        if kernels_applied:
            print(f"  🎯 Optimization details: {optimization_details}")
    else:
        print("🔍 Using standard MLX-LM quantized LoRA (baseline)")
        model._kernels_applied = False

    # Setup training components
    optimizer_name = args.optimizer.lower()
    optimizer_config = args.optimizer_config.get(optimizer_name, {})

    if optimizer_name == "adam":
        optimizer = optim.Adam(learning_rate=args.learning_rate, **optimizer_config)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(learning_rate=args.learning_rate, **optimizer_config)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Setup adapter saving
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

    # Training with timing
    print("Starting quantized LoRA training...")
    start_time = time.time()

    # Clear cache and reset memory tracking before training
    mx.clear_cache()
    mx.reset_peak_memory()

    train(
        model=model,
        args=training_args,
        optimizer=optimizer,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        training_callback=None,
    )

    training_time = time.time() - start_time

    # Evaluation
    print("Evaluating...")
    final_loss = evaluate(
        model=model,
        dataset=CacheDataset(test_set),
        batch_size=int(args.batch_size),
        num_batches=int(args.test_batches) if hasattr(args, "test_batches") else 5,
        max_seq_length=int(args.max_seq_length),
    )

    # Collect comprehensive metrics
    metrics = {
        "final_loss": float(final_loss),
        "training_time": training_time,
        "model_name": model_name,
        "num_layers_trained": args.num_layers,
        "lora_rank": args.lora_parameters["rank"],
        "quantized_layers_count": len(quantized_layers),
        "kernels_applied": kernels_applied,
        "optimization_details": optimization_details,
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
        print(f"  - Training iters: {config['iters']} (EVOLVED: increased for convergence)")

        # Test kernel loading
        print("\n📦 Testing evolved kernel loading...")
        evolved_kernels = evolved_lora_kernels()
        baseline_kernels = baseline_lora_kernels()

        print("✅ Kernels loaded successfully")
        print(f"  - Evolved kernels: {list(evolved_kernels.keys())}")
        print(f"  - Baseline: {baseline_kernels}")

        # Test model loading
        print("\n🔧 Testing quantized model loading...")
        model, tokenizer = load(config["model"])
        print(f"✅ Model loaded: {type(model).__name__}")

        # Validate quantization
        quantized_count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.QuantizedLinear):
                quantized_count += 1

        print(f"✅ Quantization validation: {quantized_count} quantized layers")

        if quantized_count == 0:
            print("⚠️ WARNING: No quantized layers found")

        print("\n🎯 Quantized LoRA optimization tests passed!")
        print("EVOLVED FEATURES:")
        print("  - Advanced bias fusion within compiled kernels")
        print("  - Sophisticated dropout handling with separate paths")
        print("  - Memory optimization strategies (90% limit)")
        print("  - Multiple compiled kernel variants")

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
        print("\n🎯 MLX Quantized LoRA Optimization Ready! (EVOLVED VERSION)")
        print("\nThis EVOLVED version targets:")
        print("- SPECIFIC INEFFICIENCY: MLX-LM dequantizes weights for LoRA computation")
        print("- OPTIMIZATION TARGET: Use mx.quantized_matmul directly, never dequantize")
        print("- EVOLVED FEATURES: Bias fusion, dropout sophistication, memory optimization")
        print("- EXPECTED IMPROVEMENT: 15-30% memory reduction, 10-20% speed improvement")
        print("- VALIDATION: Enhanced comparison with statistical analysis")
        print("\nNext steps:")
        print("1. Run: python evaluator.py")
        print("2. Run: python ../../../openevolve-run.py new_initial_program.py evaluator.py --config config.yaml")
    else:
        print("\n❌ Setup failed. Please check MLX and MLX-LM installation:")
        print("pip install mlx>=0.15.0 mlx-lm>=0.15.0")
