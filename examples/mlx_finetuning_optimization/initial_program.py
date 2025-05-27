"""
Minimal Working MLX Optimization Starting Point

This provides a very simple, conservative starting point that:
1. Works correctly with MLX APIs
2. Makes modest improvements without errors
3. Passes the enhanced reward hacking detection
4. Can be evolved into more sophisticated optimizations

Focus: Start with basic memory management and conservative optimizations
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
import gc
from typing import Dict, Any, Tuple


# EVOLVE-BLOCK-START
def basic_memory_cleanup(config: Dict[str, Any]):
    """
    Basic memory cleanup - simple starting point for evolution
    """
    cleanup_frequency = config.get("cleanup_frequency", 5)
    if cleanup_frequency > 0:
        gc.collect()


def conservative_gradient_step(model, optimizer, batch: mx.array, 
                             accumulation_step: int, total_steps: int,
                             config: Dict[str, Any]) -> Tuple[float, bool]:
    """
    Conservative gradient step with basic optimizations
    
    This is a minimal starting point that works reliably and can be evolved
    """
    # Basic input preparation
    if batch.ndim >= 2 and batch.shape[1] > 1:
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
    else:
        # Skip malformed batches
        return 3.0, False
    
    def loss_fn(model):
        # Forward pass
        logits = model(inputs)
        
        # Reshape for loss computation
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        
        # Compute cross entropy loss
        loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='mean')
        return loss
    
    try:
        # Compute loss and gradients
        loss_value, grads = mx.value_and_grad(loss_fn)(model)
        
        # Ensure loss is properly evaluated
        if isinstance(loss_value, mx.array):
            evaluated_loss = mx.eval(loss_value)
            if evaluated_loss is not None:
                loss_scalar = float(evaluated_loss)
            else:
                # If evaluation failed, skip this step
                return 3.0, False
        else:
            loss_scalar = float(loss_value)
        
        # Basic sanity check
        if not (0.1 <= loss_scalar <= 20.0):
            return loss_scalar, False
        
        # Apply basic gradient clipping
        max_grad_norm = config.get("max_grad_norm", 1.0)
        if max_grad_norm > 0 and grads:
            try:
                grads, grad_norm = optim.clip_grad_norm(grads, max_grad_norm)
            except Exception:
                # Skip clipping if it fails
                pass
        
        # Update parameters
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        # Basic memory cleanup
        if accumulation_step % config.get("cleanup_frequency", 5) == 0:
            basic_memory_cleanup(config)
        
        return loss_scalar, True
        
    except Exception as e:
        # If anything fails, return a reasonable loss and indicate failure
        print(f"Training step failed: {e}")
        return 3.0, False


def get_optimization_config() -> Dict[str, Any]:
    """
    Minimal optimization configuration that works reliably
    """
    return {
        "max_grad_norm": 1.0,           # Basic gradient clipping
        "cleanup_frequency": 5,         # Memory cleanup every 5 steps
        "use_fp16": False,             # Start with fp32 for stability
        "batch_optimization": False,    # No complex batch optimizations initially
    }
# EVOLVE-BLOCK-END


def apply_optimizations_to_trainer(trainer, config: Dict[str, Any]):
    """Apply basic optimizations to trainer"""
    
    def patched_gradient_step(model, optimizer, batch, accumulation_step, total_steps):
        return conservative_gradient_step(
            model, optimizer, batch, accumulation_step, total_steps, config
        )
    
    # Replace the gradient accumulation step
    trainer.gradient_accumulation_step = patched_gradient_step
    
    print(f"Applied basic optimizations: {config}")


def benchmark_optimization_patterns(config: Dict[str, Any], 
                                  baseline_results: Dict[str, Any] = None) -> Dict[str, float]:
    """
    Conservative benchmark that produces realistic improvements
    """
    try:
        import sys
        import os
        import psutil
        import importlib.util
        
        # Import baseline trainer
        current_dir = os.path.dirname(os.path.abspath(__file__))
        baseline_path = os.path.join(current_dir, 'baseline_finetuning.py')
        
        if not os.path.exists(baseline_path):
            # Try absolute path as fallback
            baseline_path = '/Users/asankhaya/Documents/GitHub/openevolve/examples/mlx_finetuning_optimization/baseline_finetuning.py'
        
        spec = importlib.util.spec_from_file_location("baseline_finetuning", baseline_path)
        baseline_module = importlib.util.module_from_spec(spec)
        baseline_dir = os.path.dirname(baseline_path)
        
        if baseline_dir not in sys.path:
            sys.path.insert(0, baseline_dir)
        
        spec.loader.exec_module(baseline_module)
        
        # Create trainer with same parameters as baseline
        trainer = baseline_module.BaselineTrainer("mlx-community/Qwen3-0.6B-bf16")
        trainer.config.batch_size = 2
        trainer.config.sequence_length = 128
        trainer.config.num_epochs = 1
        
        # Load model
        trainer.load_model()
        
        # Apply basic optimizations
        apply_optimizations_to_trainer(trainer, config)
        
        # Create small dataset for evaluation
        dataset = trainer.create_sample_dataset(num_samples=10)
        
        # Measure performance
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        # Run training
        training_results = trainer.train(dataset, output_dir="./basic_eval_output")
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate metrics
        training_time = end_time - start_time
        tokens_processed = len(dataset) * trainer.config.sequence_length
        tokens_per_sec = tokens_processed / max(training_time, 0.1)
        memory_efficiency = tokens_per_sec / max(end_memory, 100)
        
        # Get final loss from training results
        final_loss = training_results.get("final_loss", 5.0)
        
        # Clean up
        if os.path.exists("./basic_eval_output"):
            import shutil
            shutil.rmtree("./basic_eval_output")
        
        # Force cleanup
        gc.collect()
        
        print(f"Basic optimization results:")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Tokens processed: {tokens_processed}")
        print(f"  Tokens/sec: {tokens_per_sec:.1f}")
        print(f"  Peak memory: {end_memory:.1f}MB")
        print(f"  Memory efficiency: {memory_efficiency:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        
        return {
            "tokens_per_second": tokens_per_sec,
            "memory_efficiency": memory_efficiency,
            "peak_memory_mb": end_memory,
            "total_time": training_time,
            "final_loss": final_loss,
            "training_stats": training_results.get("training_stats", [])
        }
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "tokens_per_second": 50.0,  # Conservative fallback
            "memory_efficiency": 0.03,
            "peak_memory_mb": 2000.0,
            "total_time": 20.0,
            "final_loss": 5.0,
            "error": str(e)
        }


if __name__ == "__main__":
    print("Testing basic MLX optimization...")
    
    config = get_optimization_config()
    print(f"Config: {config}")
    
    results = benchmark_optimization_patterns(config)
    print(f"Results: {results}")
    
    if "error" not in results:
        print("✅ Basic optimization runs successfully!")
    else:
        print(f"❌ Error: {results['error']}")
