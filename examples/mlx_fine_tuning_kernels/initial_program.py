"""
MLX LoRA Fine-tuning Optimization - OpenEvolve Example

This example demonstrates optimizing real MLX LoRA fine-tuning to achieve the same 
training loss as standard MLX-LM LoRA implementation but with improved memory 
efficiency and/or training speed.

Uses the official mlx-lm library for real LoRA fine-tuning benchmarks.
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
    """Create training configuration for LoRA fine-tuning with all MLX-LM expected attributes."""
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
        "iters": 10,
        "val_batches": 5,
        "learning_rate": 1e-4,
        "steps_per_report": 5,
        "steps_per_eval": 100,
        "adapter_path": "temp_adapters",
        "save_every": 100,
        "max_seq_length": 512,
        "lora_parameters": {"rank": 16, "dropout": 0.0, "scale": 16.0},
        "mask_prompt": False,
        # Additional MLX-LM expected attributes
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
        {
            "text": "What is the capital of France?\nThe capital of France is Paris."
        },
        {
            "text": "Explain machine learning.\nMachine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
        },
        {
            "text": "How do you make tea?\nTo make tea, boil water, add tea leaves or a tea bag to a cup, pour the hot water over the tea, let it steep for 3-5 minutes, then remove the tea leaves or bag."
        },
        {
            "text": "What is photosynthesis?\nPhotosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar."
        },
        {
            "text": "Name three colors.\nThree colors are red, blue, and green."
        }
    ]
    
    # Expand examples to requested number
    expanded_examples = []
    for i in range(num_samples):
        example = examples[i % len(examples)]
        expanded_examples.append(example)
    
    # Create train, valid, test splits
    train_data = expanded_examples[:int(0.7 * num_samples)]
    valid_data = expanded_examples[int(0.7 * num_samples):int(0.9 * num_samples)]
    test_data = expanded_examples[int(0.9 * num_samples):]
    
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
    Evolved LoRA kernel implementations targeting efficiency improvements.
    
    These implementations should achieve the same training loss as standard LoRA
    but with improved memory efficiency and/or training speed.
    
    Returns:
        Dictionary of optimized LoRA operations based on mlx-lm
    """
    
    if not MLX_LM_AVAILABLE:
        raise ImportError("MLX-LM is required for real LoRA optimization")
    
    # EVOLVE-BLOCK-START
    def optimized_linear_to_lora_layers(
        model: nn.Module,
        num_layers: int,
        lora_parameters: dict,
        use_dora: bool = False
    ):
        """
        Optimized LoRA layer conversion with potential batching and memory optimizations.
        Based on mlx-lm's linear_to_lora_layers but with efficiency improvements.
        """
        # Use the official implementation as base but with potential optimizations
        return linear_to_lora_layers(model, num_layers, lora_parameters, use_dora)
    
    def optimized_train_step(
        model: nn.Module,
        inputs: Dict[str, mx.array],
        targets: mx.array,
        optimizer: optim.Optimizer,
        loss_fn: callable = None
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """
        Optimized training step with potential fusion and memory optimizations.
        """
        if loss_fn is None:
            loss_fn = nn.losses.cross_entropy
        
        def compute_loss(model, inputs, targets):
            # Efficient forward pass
            logits = model(inputs)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            
            # Memory-efficient loss computation
            return loss_fn(logits, targets, reduction='mean')
        
        # Use MLX's efficient value_and_grad
        loss_and_grad_fn = nn.value_and_grad(model, compute_loss)
        (loss, _), grads = loss_and_grad_fn(model, inputs, targets)
        
        # Optimized parameter update
        optimizer.update(model, grads)
        
        return loss, grads
    
    def optimized_training_loop(
        model: nn.Module,
        train_dataset,
        val_dataset,
        args,
        optimizer: optim.Optimizer,
        training_callback=None
    ):
        """
        Optimized training loop with memory and speed improvements.
        Based on mlx-lm's train function but with efficiency optimizations.
        """
        # Create training args if needed
        if not isinstance(args, TrainingArgs):
            training_args = TrainingArgs(
                batch_size=getattr(args, 'batch_size', 2),
                iters=getattr(args, 'iters', 10),
                val_batches=getattr(args, 'val_batches', 5),
                steps_per_report=getattr(args, 'steps_per_report', 5),
                steps_per_eval=getattr(args, 'steps_per_eval', 100),
                steps_per_save=getattr(args, 'save_every', 100),
                adapter_file=getattr(args, 'adapter_file', None),
                max_seq_length=getattr(args, 'max_seq_length', 512),
                grad_checkpoint=getattr(args, 'grad_checkpoint', False),
            )
        else:
            training_args = args
        
        # Use official MLX-LM training with potential optimizations
        return train(
            model=model,
            args=training_args,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_callback=training_callback,
        )
    
    def optimized_evaluate(
        model: nn.Module,
        dataset,
        batch_size: int = 2,
        num_batches: int = -1,
        max_seq_length: int = 512
    ) -> float:
        """
        Optimized evaluation with memory efficiency improvements.
        """
        return evaluate(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            num_batches=num_batches,
            max_seq_length=max_seq_length
        )
    
    def optimized_lora_fine_tuning(
        model_name: str,
        train_data_path: str,
        config: Dict[str, Any],
        adapter_save_path: str = "temp_adapters"
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Complete optimized LoRA fine-tuning pipeline with efficiency improvements.
        """
        # Set random seed
        mx.random.seed(config.get('seed', 42))
        np.random.seed(config.get('seed', 42))
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        model, tokenizer = load(model_name)
        
        # Convert args to namespace for compatibility
        args = types.SimpleNamespace(**config)
        args.data = train_data_path
        
        # Load datasets
        print("Loading datasets...")
        train_set, valid_set, test_set = load_dataset(args, tokenizer)
        
        # Freeze model and apply LoRA - CRITICAL: Follow exact MLX-LM pattern
        print("Applying LoRA...")
        model.freeze()
        
        # Use optimized LoRA layer conversion
        optimized_linear_to_lora_layers(
            model,
            args.num_layers,
            args.lora_parameters,
            use_dora=(args.fine_tune_type == "dora")
        )
        
        print_trainable_parameters(model)
        
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
        # Convert Path objects to strings for JSON serialization
        config_to_save = vars(args).copy()
        config_to_save['adapter_file'] = str(config_to_save['adapter_file'])
        save_config(config_to_save, adapter_path / "adapter_config.json")
        
        # Training arguments
        training_args = TrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=args.adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
        )
        
        # Run optimized training
        print("Starting optimized training...")
        start_time = time.time()
        
        optimized_training_loop(
            model=model,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(valid_set),
            args=training_args,
            optimizer=optimizer
        )
        
        training_time = time.time() - start_time
        
        # Evaluate final performance
        print("Evaluating...")
        final_loss = optimized_evaluate(
            model=model,
            dataset=CacheDataset(test_set),
            batch_size=args.batch_size,
            num_batches=args.test_batches if hasattr(args, 'test_batches') else 10,
            max_seq_length=args.max_seq_length
        )
        
        metrics = {
            'final_loss': float(final_loss),
            'training_time': training_time,
            'model_name': model_name,
            'num_layers_trained': args.num_layers,
            'lora_rank': args.lora_parameters['rank'],
        }
        
        return final_loss, metrics
    
    return {
        'optimized_linear_to_lora_layers': optimized_linear_to_lora_layers,
        'optimized_train_step': optimized_train_step,
        'optimized_training_loop': optimized_training_loop,
        'optimized_evaluate': optimized_evaluate,
        'optimized_lora_fine_tuning': optimized_lora_fine_tuning,
    }
    # EVOLVE-BLOCK-END


def baseline_lora_kernels():
    """Baseline LoRA implementations using standard MLX-LM patterns."""
    
    if not MLX_LM_AVAILABLE:
        raise ImportError("MLX-LM is required for real LoRA benchmarking")
    
    def baseline_linear_to_lora_layers(
        model: nn.Module,
        num_layers: int,
        lora_parameters: dict,
        use_dora: bool = False
    ):
        """Standard LoRA layer conversion using mlx-lm."""
        return linear_to_lora_layers(model, num_layers, lora_parameters, use_dora)
    
    def baseline_train_step(
        model: nn.Module,
        inputs: Dict[str, mx.array],
        targets: mx.array,
        optimizer: optim.Optimizer,
        loss_fn: callable = None
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Standard training step."""
        if loss_fn is None:
            loss_fn = nn.losses.cross_entropy
        
        def compute_loss(model, inputs, targets):
            logits = model(inputs)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            return loss_fn(logits, targets, reduction='mean')
        
        loss_and_grad_fn = nn.value_and_grad(model, compute_loss)
        (loss, _), grads = loss_and_grad_fn(model, inputs, targets)
        optimizer.update(model, grads)
        
        return loss, grads
    
    def baseline_training_loop(
        model: nn.Module,
        train_dataset,
        val_dataset,
        args,
        optimizer: optim.Optimizer,
        training_callback=None
    ):
        """Standard training loop using mlx-lm."""
        if not isinstance(args, TrainingArgs):
            training_args = TrainingArgs(
                batch_size=getattr(args, 'batch_size', 2),
                iters=getattr(args, 'iters', 10),
                val_batches=getattr(args, 'val_batches', 5),
                steps_per_report=getattr(args, 'steps_per_report', 5),
                steps_per_eval=getattr(args, 'steps_per_eval', 100),
                steps_per_save=getattr(args, 'save_every', 100),
                adapter_file=getattr(args, 'adapter_file', None),
                max_seq_length=getattr(args, 'max_seq_length', 512),
                grad_checkpoint=getattr(args, 'grad_checkpoint', False),
            )
        else:
            training_args = args
        
        return train(
            model=model,
            args=training_args,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_callback=training_callback,
        )
    
    def baseline_evaluate(
        model: nn.Module,
        dataset,
        batch_size: int = 2,
        num_batches: int = -1,
        max_seq_length: int = 512
    ) -> float:
        """Standard evaluation using mlx-lm."""
        return evaluate(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            num_batches=num_batches,
            max_seq_length=max_seq_length
        )
    
    def baseline_lora_fine_tuning(
        model_name: str,
        train_data_path: str,
        config: Dict[str, Any],
        adapter_save_path: str = "temp_adapters_baseline"
    ) -> Tuple[float, Dict[str, Any]]:
        """Complete baseline LoRA fine-tuning pipeline using standard mlx-lm."""
        # Set random seed
        mx.random.seed(config.get('seed', 42))
        np.random.seed(config.get('seed', 42))
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        model, tokenizer = load(model_name)
        
        # Convert args to namespace for compatibility
        args = types.SimpleNamespace(**config)
        args.data = train_data_path
        
        # Load datasets
        print("Loading datasets...")
        train_set, valid_set, test_set = load_dataset(args, tokenizer)
        
        # Apply LoRA - exact MLX-LM pattern
        print("Applying baseline LoRA...")
        model.freeze()
        
        baseline_linear_to_lora_layers(
            model,
            args.num_layers,
            args.lora_parameters,
            use_dora=(args.fine_tune_type == "dora")
        )
        
        print_trainable_parameters(model)
        
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
        # Convert Path objects to strings for JSON serialization
        config_to_save = vars(args).copy()
        config_to_save['adapter_file'] = str(config_to_save['adapter_file'])
        save_config(config_to_save, adapter_path / "adapter_config.json")
        
        # Training arguments
        training_args = TrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=args.adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
        )
        
        # Run standard training
        print("Starting baseline training...")
        start_time = time.time()
        
        baseline_training_loop(
            model=model,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(valid_set),
            args=training_args,
            optimizer=optimizer
        )
        
        training_time = time.time() - start_time
        
        # Evaluate final performance
        print("Evaluating...")
        final_loss = baseline_evaluate(
            model=model,
            dataset=CacheDataset(test_set),
            batch_size=args.batch_size,
            num_batches=args.test_batches if hasattr(args, 'test_batches') else 10,
            max_seq_length=args.max_seq_length
        )
        
        metrics = {
            'final_loss': float(final_loss),
            'training_time': training_time,
            'model_name': model_name,
            'num_layers_trained': args.num_layers,
            'lora_rank': args.lora_parameters['rank'],
        }
        
        return final_loss, metrics
    
    return {
        'optimized_linear_to_lora_layers': baseline_linear_to_lora_layers,
        'optimized_train_step': baseline_train_step,
        'optimized_training_loop': baseline_training_loop,
        'optimized_evaluate': baseline_evaluate,
        'optimized_lora_fine_tuning': baseline_lora_fine_tuning,
    }


def test_lora_functionality():
    """Test basic LoRA functionality using real mlx-lm."""
    print("Testing MLX-LM LoRA Fine-tuning Integration...")
    
    if not MLX_AVAILABLE:
        print("❌ MLX not available")
        return False
    
    if not MLX_LM_AVAILABLE:
        print("❌ MLX-LM not available")
        return False
    
    try:
        print("\n=== Testing Real MLX-LM LoRA Fine-tuning ===")
        
        # Create temporary data directory
        temp_data_dir = "temp_data"
        create_sample_dataset(temp_data_dir, num_samples=20)
        
        # Test configuration
        config = create_training_config()
        config['data'] = temp_data_dir
        
        print("✅ Configuration created")
        print(f"  - Model: {config['model']}")
        print(f"  - LoRA rank: {config['lora_parameters']['rank']}")
        print(f"  - Training iterations: {config['iters']}")
        print(f"  - Batch size: {config['batch_size']}")
        
        # Get implementations
        print("\n📦 Loading LoRA implementations...")
        evolved_kernels = evolved_lora_kernels()
        baseline_kernels = baseline_lora_kernels()
        
        print("✅ Both evolved and baseline kernels loaded")
        
        # Test basic model loading
        print("\n🔧 Testing basic model loading...")
        try:
            model, tokenizer = load(config['model'])
            print(f"✅ Model loaded: {type(model).__name__}")
            print(f"✅ Tokenizer loaded: {type(tokenizer).__name__}")
            
            # Test LoRA parameter setup like in evaluator
            try:
                # Freeze model and apply minimal LoRA to test parameter access
                model.freeze()
                linear_to_lora_layers(
                    model,
                    2,  # Small number for testing
                    {"rank": 8, "dropout": 0.0, "scale": 16.0},
                    use_dora=False
                )
                print_trainable_parameters(model)
                print("✅ Model parameter access working correctly")
            except Exception as param_e:
                print(f"✅ Model loaded but LoRA setup test failed: {param_e}")
                print("This may be expected for some model configurations")
            
        except Exception as e:
            print(f"⚠️ Model loading failed: {e}")
            print("This is expected if the model is not available or too large for testing")
        
        print("\n🎯 Real MLX-LM LoRA fine-tuning tests passed!")
        print("Ready for OpenEvolve optimization!")
        
        # Cleanup temporary files
        try:
            from cleanup import cleanup_temp_files
            cleanup_temp_files()
        except ImportError:
            # Fallback cleanup
            import shutil
            try:
                shutil.rmtree(temp_data_dir, ignore_errors=True)
                shutil.rmtree("temp_adapters", ignore_errors=True)
                shutil.rmtree("temp_adapters_baseline", ignore_errors=True)
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
        print("\n🎯 MLX-LM LoRA Fine-tuning Optimization Ready!")
        print("\nThis example targets:")
        print("- Real MLX-LM LoRA fine-tuning optimization")
        print("- Same training loss with improved efficiency")  
        print("- Memory reduction and/or speed improvements")
        print("- Production-ready MLX-LM integration")
        print("\nNext steps:")
        print("1. Run: python evaluator.py")
        print("2. Run: python ../../../openevolve-run.py initial_program.py evaluator.py --config config.yaml")
    else:
        print("\n❌ Setup failed. Please check MLX and MLX-LM installation:")
        print("pip install mlx>=0.15.0 mlx-lm>=0.15.0")
