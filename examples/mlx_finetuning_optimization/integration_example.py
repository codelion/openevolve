#!/usr/bin/env python3
"""
Example: Integrating MLX Optimizations into Existing Code

This example shows how to integrate evolved MLX optimization patterns
into your existing fine-tuning code with minimal changes.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load

# Import the optimization patch
from mlx_optimization_patch import mlx_optimizations, apply_optimizations


def existing_finetuning_function():
    """
    Example of existing MLX fine-tuning code that users might have.
    This represents typical fine-tuning logic before optimization.
    """
    print("🔧 Original Fine-tuning Function")
    
    # Load model and tokenizer
    model, tokenizer = load("mlx-community/Qwen3-0.6B-bf16")
    
    # Setup training
    optimizer = optim.AdamW(learning_rate=5e-5)
    
    # Create some sample data
    texts = [
        "What is machine learning?",
        "Explain neural networks.",
        "How does fine-tuning work?"
    ]
    
    # Simple training loop
    for epoch in range(2):
        for text in texts:
            tokens = mx.array([tokenizer.encode(text)])
            
            def loss_fn(model):
                logits = model(tokens[:, :-1])
                targets = tokens[:, 1:]
                return nn.losses.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1)
                )
            
            loss, grads = mx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            print(f"Epoch {epoch}, Loss: {float(loss):.4f}")
    
    print("Original training complete!")
    return model


def optimized_finetuning_function():
    """
    Same fine-tuning function but with MLX optimizations applied.
    Only requires adding the context manager!
    """
    print("⚡ Optimized Fine-tuning Function")
    
    # The magic: wrap your existing code with optimizations
    with mlx_optimizations():
        # Your existing fine-tuning code goes here unchanged
        model, tokenizer = load("mlx-community/Qwen3-0.6B-bf16")
        
        # Setup training (same as before)
        optimizer = optim.AdamW(learning_rate=5e-5)
        
        # Create some sample data (same as before)
        texts = [
            "What is machine learning?", 
            "Explain neural networks.",
            "How does fine-tuning work?"
        ]
        
        # Training loop (same as before, but now optimized!)
        for epoch in range(2):
            for text in texts:
                tokens = mx.array([tokenizer.encode(text)])
                
                def loss_fn(model):
                    logits = model(tokens[:, :-1])
                    targets = tokens[:, 1:]
                    return nn.losses.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        targets.reshape(-1)
                    )
                
                loss, grads = mx.value_and_grad(loss_fn)(model)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                
                print(f"Epoch {epoch}, Loss: {float(loss):.4f}")
        
        print("Optimized training complete!")
        return model


class ExistingTrainerClass:
    """
    Example of an existing trainer class that users might have.
    Shows how to apply optimizations to class-based training.
    """
    
    def __init__(self, model_name="mlx-community/Qwen3-0.6B-bf16"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.optimizer = None
    
    def load_model(self):
        """Load model and tokenizer"""
        self.model, self.tokenizer = load(self.model_name)
        self.optimizer = optim.AdamW(learning_rate=5e-5)
        print(f"Loaded model: {self.model_name}")
    
    def prepare_batch(self, texts):
        """Prepare a batch of texts for training"""
        tokenized = []
        max_length = 0
        
        for text in texts:
            tokens = self.tokenizer.encode(text)
            tokenized.append(tokens)
            max_length = max(max_length, len(tokens))
        
        # Pad sequences
        padded = []
        for tokens in tokenized:
            if len(tokens) < max_length:
                # Handle different tokenizer types
                pad_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else self.tokenizer.eos_token_id
                tokens = tokens + [pad_id] * (max_length - len(tokens))
            padded.append(tokens)
        
        return mx.array(padded)
    
    def train_step(self, batch):
        """Single training step"""
        def loss_fn(model):
            logits = self.model(batch[:, :-1])
            targets = batch[:, 1:]
            return nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1)
            )
        
        loss, grads = mx.value_and_grad(loss_fn)(self.model)
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)
        
        return float(loss)
    
    def train(self, texts, epochs=2):
        """Training loop"""
        print(f"Training on {len(texts)} samples for {epochs} epochs")
        
        if self.model is None:
            self.load_model()
        
        for epoch in range(epochs):
            batch = self.prepare_batch(texts)
            loss = self.train_step(batch)
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
        
        print("Training complete!")


def example_class_based_optimization():
    """
    Example of applying optimizations to an existing trainer class
    """
    print("🏗️  Class-based Optimization Example")
    
    # Create your existing trainer
    trainer = ExistingTrainerClass()
    
    # Apply optimizations to the trainer
    apply_optimizations(trainer)
    print("✅ Optimizations applied to trainer")
    
    # Use trainer as normal - optimizations are now active
    sample_texts = [
        "### Instruction:\nWhat is artificial intelligence?\n\n### Response:\nAI is...",
        "### Instruction:\nExplain machine learning.\n\n### Response:\nMachine learning is...",
        "### Instruction:\nWhat are neural networks?\n\n### Response:\nNeural networks are..."
    ]
    
    trainer.train(sample_texts, epochs=2)
    return trainer


def example_custom_optimization_config():
    """
    Example of using custom optimization configurations
    """
    print("⚙️  Custom Configuration Example")
    
    from mlx_optimization_patch import load_optimizations
    
    # Load optimizations and inspect configuration
    patch = load_optimizations()
    config = patch.get_config()
    
    print("Current optimization configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # You could modify configuration here if needed
    # config["attention_chunk_size"] = 1024
    # config["use_fp16_compute"] = False
    
    print("\nUsing optimizations with current config...")
    
    with mlx_optimizations():
        # Your training code here will use the configuration
        print("Training with optimized patterns...")


def performance_comparison_example():
    """
    Example of comparing performance before and after optimization
    """
    print("📊 Performance Comparison Example")
    
    import time
    import psutil
    import os
    
    def measure_performance(func, name):
        """Measure execution time and memory usage"""
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        try:
            result = func()
            success = True
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            result = None
            success = False
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"\n{name} Results:")
        print(f"  Success: {success}")
        print(f"  Time: {end_time - start_time:.2f}s")
        print(f"  Memory delta: {end_memory - start_memory:.1f} MB")
        print(f"  Peak memory: {end_memory:.1f} MB")
        
        return {
            "success": success,
            "time": end_time - start_time,
            "memory_delta": end_memory - start_memory,
            "peak_memory": end_memory
        }
    
    # Compare baseline vs optimized
    print("Running baseline training...")
    baseline_results = measure_performance(existing_finetuning_function, "Baseline")
    
    print("\nRunning optimized training...")
    optimized_results = measure_performance(optimized_finetuning_function, "Optimized")
    
    # Calculate improvements
    if baseline_results["success"] and optimized_results["success"]:
        time_improvement = (baseline_results["time"] - optimized_results["time"]) / baseline_results["time"]
        memory_improvement = (baseline_results["peak_memory"] - optimized_results["peak_memory"]) / baseline_results["peak_memory"]
        
        print(f"\n🎯 Performance Improvements:")
        print(f"  Time: {time_improvement:+.1%}")
        print(f"  Memory: {memory_improvement:+.1%}")


def main():
    """Main example function"""
    print("🚀 MLX Fine-tuning Optimization Integration Examples")
    print("=" * 60)
    
    examples = [
        ("Context Manager", optimized_finetuning_function),
        ("Class-based Optimization", example_class_based_optimization),
        ("Custom Configuration", example_custom_optimization_config),
        ("Performance Comparison", performance_comparison_example),
    ]
    
    for name, example_func in examples:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            example_func()
        except Exception as e:
            print(f"❌ Example failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("✅ All integration examples completed!")
    print("\n💡 Key takeaways:")
    print("  1. Use 'with mlx_optimizations():' for drop-in optimization")
    print("  2. Use 'apply_optimizations(trainer)' for class-based trainers")
    print("  3. Optimizations are automatically loaded from evolved patterns")
    print("  4. No changes needed to your existing training logic!")


if __name__ == "__main__":
    main()
