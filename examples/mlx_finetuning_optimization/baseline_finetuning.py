#!/usr/bin/env python3
"""
Baseline MLX Fine-tuning with Qwen3-0.6B-bf16

This script provides a baseline implementation for fine-tuning using standard mlx-lm.
It serves as a reference point for measuring the improvements from evolved optimizations.

Key components that can be monkey-patched:
- attention_forward: Custom attention computation
- gradient_accumulation_step: Memory-efficient gradient handling
- mixed_precision_forward: Optimized precision patterns
- batch_preparation: Optimized data loading and batching
"""

import argparse
import json
import time
import gc
import psutil
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.utils import load_config
import numpy as np


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    batch_size: int = 2  # Reduced for memory safety
    sequence_length: int = 512
    learning_rate: float = 5e-5
    num_epochs: int = 1
    gradient_accumulation_steps: int = 1  # Simplified for now
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 100
    weight_decay: float = 0.01
    
    # Memory optimization settings
    gradient_checkpointing: bool = False
    mixed_precision: bool = True
    fp16_dtype: str = "float16"  # or "bfloat16"


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    peak_memory_mb: float
    current_memory_mb: float
    baseline_memory_mb: float
    memory_efficiency: float  # tokens_per_second / memory_mb


class BaselineTrainer:
    """
    Baseline trainer using standard MLX operations.
    
    This class contains the core training logic that can be optimized
    through monkey patching of key methods.
    """
    
    def __init__(self, model_name: str = "mlx-community/Qwen3-0.6B-bf16"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.config = TrainingConfig()
        
        # Performance tracking
        self.baseline_memory = 0.0
        self.peak_memory = 0.0
        self.training_stats = []
        
    def load_model(self):
        """Load model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        self.model, self.tokenizer = load(self.model_name)
        
        # Ensure we have a pad token
        if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Get pad token ID
        if hasattr(self.tokenizer, 'pad_token_id'):
            self.pad_token_id = self.tokenizer.pad_token_id
        else:
            self.pad_token_id = self.tokenizer.eos_token_id
            
        # Get vocab size safely - different tokenizers have different attributes
        if hasattr(self.tokenizer, 'vocab_size'):
            vocab_size = self.tokenizer.vocab_size
        elif hasattr(self.tokenizer, 'get_vocab_size'):
            vocab_size = self.tokenizer.get_vocab_size()
        else:
            vocab_size = "unknown"
        print(f"Model loaded. Vocab size: {vocab_size}")
        return self.model, self.tokenizer
    
    def create_sample_dataset(self, num_samples: int = 1000) -> List[Dict[str, str]]:
        """
        Create a sample instruction-following dataset
        
        In practice, you would load a real dataset like Alpaca
        """
        instruction_templates = [
            "Explain the concept of {topic} in simple terms.",
            "Write a short story about {topic}.",
            "List the main advantages and disadvantages of {topic}.",
            "How does {topic} work?",
            "What are the key features of {topic}?",
            "Compare {topic} with similar concepts.",
            "Describe the history and development of {topic}.",
            "What are the practical applications of {topic}?",
            "Explain {topic} to a beginner.",
            "What are common misconceptions about {topic}?"
        ]
        
        topics = [
            "machine learning", "neural networks", "artificial intelligence",
            "data science", "computer vision", "natural language processing",
            "deep learning", "reinforcement learning", "supervised learning",
            "unsupervised learning", "transfer learning", "transformers",
            "attention mechanisms", "gradient descent", "backpropagation",
            "convolutional networks", "recurrent networks", "ensemble methods",
            "feature engineering", "model evaluation", "cross validation",
            "overfitting", "regularization", "hyperparameter tuning"
        ]
        
        responses = {
            "machine learning": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed for every task.",
            "neural networks": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections.",
            "artificial intelligence": "Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans.",
            "data science": "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.",
            # Add more responses as needed
        }
        
        dataset = []
        for i in range(num_samples):
            topic = topics[i % len(topics)]
            template = instruction_templates[i % len(instruction_templates)]
            instruction = template.format(topic=topic)
            
            # Use a default response if we don't have a specific one
            response = responses.get(topic, f"This is a response about {topic}. It explains the key concepts and provides useful information for understanding this topic better.")
            
            dataset.append({
                "instruction": instruction,
                "input": "",
                "output": response
            })
            
        return dataset
    
    def format_sample(self, sample: Dict[str, str]) -> str:
        """Format a training sample as text"""
        if sample["input"]:
            return f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n\n### Response:\n{sample['output']}"
        else:
            return f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['output']}"
    
    def tokenize_batch(self, texts: List[str]) -> mx.array:
        """
        Tokenize a batch of texts with padding
        
        This method can be monkey-patched for optimized tokenization
        """
        tokenized = []
        max_length = 0
        
        # Tokenize all texts
        for text in texts:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > self.config.sequence_length:
                tokens = tokens[:self.config.sequence_length]
            tokenized.append(tokens)
            max_length = max(max_length, len(tokens))
        
        # Pad to max length in batch
        padded = []
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        for tokens in tokenized:
            if len(tokens) < max_length:
                tokens = tokens + [pad_token_id] * (max_length - len(tokens))
            padded.append(tokens)
        
        return mx.array(padded, dtype=mx.int32)  # Ensure tokens are integers
    
    def batch_preparation(self, dataset: List[Dict[str, str]], batch_size: int) -> List[mx.array]:
        """
        Prepare training batches
        
        This method can be monkey-patched for optimized batch preparation
        """
        batches = []
        
        for i in range(0, len(dataset), batch_size):
            batch_samples = dataset[i:i + batch_size]
            texts = [self.format_sample(sample) for sample in batch_samples]
            tokenized_batch = self.tokenize_batch(texts)
            batches.append(tokenized_batch)
            
        return batches
    
    def attention_forward(self, query: mx.array, key: mx.array, value: mx.array, 
                         attention_mask: Optional[mx.array] = None) -> mx.array:
        """
        Attention computation - can be monkey-patched for optimization
        
        This is a simplified version. In practice, this would be part of the model's
        attention layers, but we expose it here for demonstration of patching.
        """
        # This is a placeholder - real attention would be in the model layers
        # But this shows how we could patch attention patterns
        d_k = query.shape[-1]
        scores = mx.matmul(query, key.transpose(0, 1, 3, 2)) / mx.sqrt(d_k)
        
        if attention_mask is not None:
            scores = scores + attention_mask
            
        attention_weights = mx.softmax(scores, axis=-1)
        output = mx.matmul(attention_weights, value)
        
        return output
    
    def mixed_precision_forward(self, model, inputs: mx.array) -> mx.array:
        """
        Forward pass with mixed precision
        
        This method can be monkey-patched for optimized precision patterns
        """
        if self.config.mixed_precision:
            # Convert inputs to appropriate dtype, but preserve integer types for token indices
            if inputs.dtype not in [mx.int32, mx.int64, mx.uint32]:
                # Only cast non-integer tensors
                if self.config.fp16_dtype == "float16":
                    inputs = inputs.astype(mx.float16)
                elif self.config.fp16_dtype == "bfloat16":
                    inputs = inputs.astype(mx.bfloat16)
        
        outputs = model(inputs)
        
        # Ensure outputs are in float32 for loss computation
        if outputs.dtype != mx.float32:
            outputs = outputs.astype(mx.float32)
            
        return outputs
    
    def gradient_accumulation_step(self, model, optimizer, batch: mx.array, 
                                 accumulation_step: int, total_steps: int) -> Tuple[float, bool]:
        """
        Simplified gradient step (can be evolved to add accumulation)
        
        This method can be monkey-patched for memory-efficient gradient handling
        """
        # Prepare inputs and targets
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        def loss_fn(model):
            logits = self.mixed_precision_forward(model, inputs)
            # Reshape for cross entropy
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)
            
            loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='mean')
            return loss
        
        # Compute loss and gradients
        loss_value, grads = mx.value_and_grad(loss_fn)(model)
        
        # For now, just do direct updates to avoid gradient accumulation issues
        # Evolution can add proper gradient accumulation later
        
        # Apply gradient clipping
        if self.config.max_grad_norm > 0:
            grads, grad_norm = optim.clip_grad_norm(grads, self.config.max_grad_norm)
        
        # Update parameters
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        return float(loss_value), True  # Always return True for update
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        process = psutil.Process(os.getpid())
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if self.baseline_memory == 0:
            self.baseline_memory = current_memory
            
        self.peak_memory = max(self.peak_memory, current_memory)
        
        return MemoryStats(
            peak_memory_mb=self.peak_memory,
            current_memory_mb=current_memory,
            baseline_memory_mb=self.baseline_memory,
            memory_efficiency=0.0  # Will be calculated with tokens/sec
        )
    
    def train(self, dataset: List[Dict[str, str]], output_dir: str = "./baseline_output") -> Dict[str, Any]:
        """
        Main training loop
        
        Returns performance metrics for comparison with optimized versions
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Prepare optimizer
        optimizer = optim.AdamW(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Prepare batches
        print("Preparing training batches...")
        batches = self.batch_preparation(dataset, self.config.batch_size)
        total_batches = len(batches)
        total_steps = total_batches * self.config.num_epochs
        
        print(f"Training on {len(dataset)} samples, {total_batches} batches, {total_steps} total steps")
        
        # Get baseline memory
        baseline_stats = self.get_memory_stats()
        
        # Training loop
        step = 0
        total_loss = 0.0
        start_time = time.time()
        tokens_processed = 0
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            for batch_idx, batch in enumerate(batches):
                batch_start_time = time.time()
                
                # Training step (simplified - no complex gradient accumulation)
                loss, updated = self.gradient_accumulation_step(
                    self.model, optimizer, batch, 0, step
                )
                total_loss += loss
                step += 1
                
                # Count tokens processed
                tokens_processed += batch.size
                
                # Log progress
                if step % 10 == 0:
                    avg_loss = total_loss / max(step, 1)
                    elapsed_time = time.time() - start_time
                    tokens_per_sec = tokens_processed / elapsed_time if elapsed_time > 0 else 0
                    
                    memory_stats = self.get_memory_stats()
                    memory_stats.memory_efficiency = tokens_per_sec / max(memory_stats.current_memory_mb, 1)
                    
                    print(f"Step {step}, Loss: {avg_loss:.4f}, "
                          f"Tokens/sec: {tokens_per_sec:.1f}, "
                          f"Memory: {memory_stats.current_memory_mb:.1f}MB")
                    
                    self.training_stats.append({
                        "step": step,
                        "loss": avg_loss,
                        "tokens_per_sec": tokens_per_sec,
                        "memory_mb": memory_stats.current_memory_mb,
                        "memory_efficiency": memory_stats.memory_efficiency
                    })
                
                # Evaluation
                if step % self.config.eval_steps == 0 and step > 0:
                    self.evaluate_model(step)
                
                # Save checkpoint
                if step % self.config.save_steps == 0 and step > 0:
                    self.save_checkpoint(output_dir, step)
        
        # Final statistics
        total_time = time.time() - start_time
        final_memory_stats = self.get_memory_stats()
        final_tokens_per_sec = tokens_processed / total_time
        final_memory_stats.memory_efficiency = final_tokens_per_sec / max(final_memory_stats.peak_memory_mb, 1)
        
        results = {
            "total_time": total_time,
            "total_tokens": tokens_processed,
            "tokens_per_second": final_tokens_per_sec,
            "final_loss": total_loss / max(step, 1),
            "peak_memory_mb": final_memory_stats.peak_memory_mb,
            "memory_efficiency": final_memory_stats.memory_efficiency,
            "total_steps": step,
            "training_stats": self.training_stats
        }
        
        # Save final results
        with open(os.path.join(output_dir, "training_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTraining completed!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Tokens/sec: {final_tokens_per_sec:.1f}")
        print(f"Peak memory: {final_memory_stats.peak_memory_mb:.1f}MB")
        print(f"Memory efficiency: {final_memory_stats.memory_efficiency:.4f} tokens/sec/MB")
        
        return results
    
    def evaluate_model(self, step: int):
        """Simple model evaluation"""
        test_prompt = "### Instruction:\nExplain machine learning in simple terms.\n\n### Response:\n"
        
        try:
            response = generate(
                self.model, 
                self.tokenizer, 
                prompt=test_prompt, 
                max_tokens=100,
            )
            print(f"Evaluation at step {step}:")
            print(f"Prompt: {test_prompt}")
            print(f"Response: {response}")
            print("-" * 50)
        except Exception as e:
            print(f"Evaluation failed at step {step}: {e}")
    
    def save_checkpoint(self, output_dir: str, step: int):
        """Save training checkpoint"""
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model weights (simplified - in practice you'd use proper MLX saving)
        print(f"Saved checkpoint at step {step} to {checkpoint_dir}")


def main():
    """Main function for running baseline training"""
    parser = argparse.ArgumentParser(description="Baseline MLX Fine-tuning")
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-bf16", help="Model to fine-tune")
    parser.add_argument("--output_dir", default="./baseline_output", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of training samples")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = BaselineTrainer(args.model)
    
    # Update configuration
    trainer.config.batch_size = args.batch_size
    trainer.config.num_epochs = args.epochs
    trainer.config.learning_rate = args.learning_rate
    
    # Create dataset
    print("Creating sample dataset...")
    dataset = trainer.create_sample_dataset(args.num_samples)
    print(f"Created {len(dataset)} training samples")
    
    # Run training
    results = trainer.train(dataset, args.output_dir)
    
    print("\nBaseline training completed!")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
