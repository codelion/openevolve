"""
Real-World LLM Fine-tuning Evaluator for MLX Optimization

This evaluator measures performance improvements on actual LLM fine-tuning
using mlx-community/SmolLM2-135M-Instruct, providing realistic metrics
for production ML workloads.
"""

import importlib.util
import time
import traceback
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import gc
import os
import tempfile
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Try importing mlx-lm components
try:
    from mlx_lm import load, generate
    HAS_MLX_LM = True
except ImportError:
    print("⚠️  mlx-lm not found. Install with: pip install mlx-lm")
    HAS_MLX_LM = False


def create_fine_tuning_dataset(num_samples: int = 50) -> List[Dict]:
    """Create a simple instruction-following dataset for fine-tuning evaluation"""
    
    # Simple instruction-response pairs for fine-tuning
    instructions = [
        {"instruction": "What is the capital of France?", "response": "The capital of France is Paris."},
        {"instruction": "Explain what machine learning is.", "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."},
        {"instruction": "How do you make a paper airplane?", "response": "To make a paper airplane: 1) Fold a sheet of paper in half lengthwise, 2) Unfold and fold the top corners to the center line, 3) Fold the angled edges to the center line again, 4) Fold the plane in half, 5) Create wings by folding each side down."},
        {"instruction": "What are the primary colors?", "response": "The primary colors are red, blue, and yellow. These colors cannot be created by mixing other colors together."},
        {"instruction": "Describe the water cycle.", "response": "The water cycle consists of evaporation, condensation, precipitation, and collection. Water evaporates from oceans and lakes, forms clouds, falls as rain or snow, and returns to water bodies."},
        {"instruction": "What is photosynthesis?", "response": "Photosynthesis is the process by which plants use sunlight, carbon dioxide, and water to produce glucose and oxygen. This process is essential for life on Earth."},
        {"instruction": "How do you calculate the area of a circle?", "response": "The area of a circle is calculated using the formula A = πr², where A is the area, π (pi) is approximately 3.14159, and r is the radius of the circle."},
        {"instruction": "What causes seasons on Earth?", "response": "Seasons are caused by the tilt of Earth's axis as it orbits the sun. This tilt causes different parts of Earth to receive varying amounts of direct sunlight throughout the year."},
    ]
    
    # Repeat and shuffle to create desired number of samples
    dataset = []
    for i in range(num_samples):
        sample = instructions[i % len(instructions)]
        # Format as instruction-following format
        formatted_sample = {
            "text": f"<|im_start|>user\n{sample['instruction']}<|im_end|>\n<|im_start|>assistant\n{sample['response']}<|im_end|>"
        }
        dataset.append(formatted_sample)
    
    return dataset


def tokenize_dataset(dataset: List[Dict], tokenizer, max_length: int = 512) -> List[mx.array]:
    """Tokenize the dataset for training"""
    
    tokenized_samples = []
    
    for sample in dataset:
        # Tokenize the text
        tokens = tokenizer.encode(sample["text"])
        
        # Truncate or pad to max_length
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Convert to MLX array
        token_array = mx.array(tokens, dtype=mx.int32)
        tokenized_samples.append(token_array)
    
    return tokenized_samples


def create_batches(tokenized_samples: List[mx.array], batch_size: int = 4, seq_length: int = 512) -> List[Tuple[mx.array, mx.array]]:
    """Create training batches with proper input/target formatting"""
    
    batches = []
    
    for i in range(0, len(tokenized_samples), batch_size):
        batch_samples = tokenized_samples[i:i + batch_size]
        
        # Pad all samples in batch to same length
        batch_tokens = []
        for sample in batch_samples:
            if len(sample) < seq_length:
                # Pad with tokenizer pad token (usually 0)
                padded = mx.concatenate([sample, mx.zeros(seq_length - len(sample), dtype=mx.int32)])
            else:
                padded = sample[:seq_length]
            batch_tokens.append(padded)
        
        # Stack into batch
        if len(batch_tokens) == batch_size:
            batch_tensor = mx.stack(batch_tokens)
            
            # Create input/target pairs (shift by 1 for next-token prediction)
            inputs = batch_tensor[:, :-1]
            targets = batch_tensor[:, 1:]
            
            batches.append((inputs, targets))
    
    return batches


def evaluate_real_llm_finetuning(program_path: str) -> Dict:
    """
    Evaluate MLX optimization performance on real LLM fine-tuning
    
    This function loads SmolLM2-135M-Instruct and measures the performance
    improvement during actual fine-tuning with the evolved optimizations.
    """
    
    if not HAS_MLX_LM:
        return {
            "training_speedup": 0.0,
            "memory_efficiency": 0.0,
            "combined_score": 0.0,
            "error": "mlx-lm not available"
        }
    
    try:
        # Load the evolved program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Check required functions exist
        required_functions = ["get_device_info", "choose_tile_size", "optimized_matmul"]
        for func_name in required_functions:
            if not hasattr(program, func_name):
                return {
                    "training_speedup": 0.0,
                    "memory_efficiency": 0.0,
                    "combined_score": 0.0,
                    "error": f"Missing function: {func_name}"
                }
        
        print("🔄 Loading SmolLM2-135M-Instruct...")
        
        # Load the real model
        try:
            model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
            print("✅ Model loaded successfully")
        except Exception as e:
            return {
                "training_speedup": 0.0,
                "memory_efficiency": 0.0, 
                "combined_score": 0.0,
                "error": f"Failed to load model: {str(e)}"
            }
        
        # Create fine-tuning dataset
        print("📝 Creating fine-tuning dataset...")
        dataset = create_fine_tuning_dataset(num_samples=20)  # Small dataset for evaluation
        tokenized_samples = tokenize_dataset(dataset, tokenizer, max_length=256)
        batches = create_batches(tokenized_samples, batch_size=2, seq_length=256)  # Small batch for memory
        
        if len(batches) == 0:
            return {
                "training_speedup": 0.0,
                "memory_efficiency": 0.0,
                "combined_score": 0.0,
                "error": "No training batches created"
            }
        
        print(f"📊 Created {len(batches)} training batches")
        
        # Test baseline performance (standard MLX)
        print("🔬 Testing baseline performance...")
        baseline_results = benchmark_finetuning_performance(
            model, tokenizer, batches, program, use_optimization=False
        )
        
        if "error" in baseline_results:
            return {
                "training_speedup": 0.0,
                "memory_efficiency": 0.0,
                "combined_score": 0.0,
                "error": f"Baseline failed: {baseline_results['error']}"
            }
        
        # Test optimized performance
        print("⚡ Testing optimized performance...")
        optimized_results = benchmark_finetuning_performance(
            model, tokenizer, batches, program, use_optimization=True
        )
        
        if "error" in optimized_results:
            return {
                "training_speedup": 0.0,
                "memory_efficiency": 0.0,
                "combined_score": 0.0,
                "error": f"Optimized failed: {optimized_results['error']}"
            }
        
        # Calculate performance metrics
        baseline_time = baseline_results["avg_step_time"]
        optimized_time = optimized_results["avg_step_time"]
        
        baseline_memory = baseline_results.get("peak_memory", 0)
        optimized_memory = optimized_results.get("peak_memory", 0)
        
        # Training speedup
        training_speedup = baseline_time / optimized_time if optimized_time > 0 else 0.0
        
        # Memory efficiency (lower memory usage is better)
        memory_efficiency = baseline_memory / max(optimized_memory, 1) if optimized_memory > 0 else 1.0
        
        # Combined score (weight speedup more heavily than memory)
        combined_score = 0.8 * training_speedup + 0.2 * memory_efficiency
        
        # Bonus for significant improvements
        if training_speedup > 1.05:  # >5% speedup
            combined_score *= 1.2
        elif training_speedup > 1.02:  # >2% speedup
            combined_score *= 1.1
        
        print(f"📈 Results: {training_speedup:.3f}x speedup, {memory_efficiency:.3f}x memory efficiency")
        
        return {
            "training_speedup": float(training_speedup),
            "memory_efficiency": float(memory_efficiency),
            "baseline_step_time": float(baseline_time),
            "optimized_step_time": float(optimized_time),
            "baseline_memory": float(baseline_memory),
            "optimized_memory": float(optimized_memory),
            "combined_score": float(combined_score),
            "optimizations_applied": int(optimized_results.get("optimizations_applied", 0)),
            "test_successful": True,
            "model_name": "SmolLM2-135M-Instruct"
        }
        
    except Exception as e:
        print(f"💥 Evaluation failed: {str(e)}")
        traceback.print_exc()
        return {
            "training_speedup": 0.0,
            "memory_efficiency": 0.0,
            "combined_score": 0.0,
            "error": f"Evaluation exception: {str(e)}"
        }


def benchmark_finetuning_performance(
    model, 
    tokenizer, 
    batches: List[Tuple[mx.array, mx.array]], 
    program, 
    use_optimization: bool = False,
    num_steps: int = 5
) -> Dict:
    """
    Benchmark fine-tuning performance with or without optimization
    """
    
    try:
        # Store original matmul
        original_matmul = mx.matmul
        optimization_count = 0
        
        if use_optimization:
            # Get device info
            device_info = program.get_device_info()
            
            # Create optimized matmul function
            def create_optimized_matmul():
                def optimized_matmul_with_tracking(A, B):
                    nonlocal optimization_count
                    
                    # Same logic as mlx_lm_openevolve.py
                    if (len(A.shape) == 2 and len(B.shape) == 2 and 
                        A.shape[0] * A.shape[1] * B.shape[1] > 2**18):  # Lower threshold for real models
                        
                        M, K1 = A.shape
                        K2, N = B.shape
                        
                        if K1 == K2:
                            try:
                                tile_M, tile_N, tile_K = program.choose_tile_size(M, N, K1, device_info)
                                if tile_M > 0 and tile_N > 0 and tile_K > 0:
                                    optimization_count += 1
                                    return program.optimized_matmul(A, B, tile_M, tile_N, tile_K)
                            except Exception:
                                pass  # Fall back to original
                    
                    return original_matmul(A, B)
                return optimized_matmul_with_tracking
            
            mx.matmul = create_optimized_matmul()
        
        # Create optimizer for fine-tuning
        optimizer = optim.Adam(learning_rate=1e-5)  # Conservative LR for fine-tuning
        
        # Loss function for causal language modeling
        def loss_fn(model, inputs, targets):
            logits = model(inputs)
            batch_size, seq_len, vocab_size = logits.shape
            
            # Reshape for cross-entropy
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1)
            
            # Simple cross-entropy without masking to avoid boolean indexing issues
            # MLX doesn't support boolean indexing, so we'll compute loss on all tokens
            return nn.losses.cross_entropy(logits_flat, targets_flat, reduction='mean')
        
        # Gradient function
        value_and_grad_fn = mx.value_and_grad(loss_fn)
        
        # Memory tracking
        def get_memory_usage():
            # Simple memory estimation based on array sizes
            total_memory = 0
            try:
                for param in model.parameters():
                    if hasattr(param, 'shape'):
                        # Calculate memory usage: shape -> total elements -> bytes
                        total_elements = 1
                        for dim in param.shape:
                            total_elements *= dim
                        total_memory += total_elements * 4  # Assume 4 bytes per float32
            except Exception:
                # Fallback to simple estimation
                total_memory = 64 * 1024 * 1024  # 64MB default
            return total_memory / (1024 * 1024)  # MB
        
        initial_memory = get_memory_usage()
        peak_memory = initial_memory
        
        # Warmup
        if len(batches) > 0:
            inputs, targets = batches[0]
            for _ in range(2):
                try:
                    loss, grads = value_and_grad_fn(model, inputs, targets)
                    optimizer.update(model, grads)
                    mx.eval(model.parameters(), optimizer.state, loss)
                    
                    # Update peak memory
                    current_memory = get_memory_usage()
                    peak_memory = max(peak_memory, current_memory)
                    
                except Exception as e:
                    print(f"Warmup step failed: {e}")
                    break
        
        # Benchmark training steps
        step_times = []
        losses = []
        
        for step in range(min(num_steps, len(batches))):
            inputs, targets = batches[step % len(batches)]
            
            start_time = time.perf_counter()
            
            try:
                # Forward and backward pass
                loss, grads = value_and_grad_fn(model, inputs, targets)
                
                # Parameter update
                optimizer.update(model, grads)
                
                # Ensure computation is complete
                mx.eval(model.parameters(), optimizer.state, loss)
                
                end_time = time.perf_counter()
                step_time = end_time - start_time
                step_times.append(step_time)
                losses.append(float(loss))
                
                # Update peak memory
                current_memory = get_memory_usage()
                peak_memory = max(peak_memory, current_memory)
                
            except Exception as e:
                print(f"Training step {step} failed: {e}")
                break
        
        # Restore original matmul
        mx.matmul = original_matmul
        
        if len(step_times) == 0:
            return {"error": "No successful training steps"}
        
        # Calculate metrics
        avg_step_time = np.median(step_times)
        final_loss = losses[-1] if losses else float('inf')
        
        return {
            "avg_step_time": avg_step_time,
            "final_loss": final_loss,
            "peak_memory": peak_memory,
            "optimizations_applied": optimization_count,
            "successful_steps": len(step_times),
            "step_times": step_times
        }
        
    except Exception as e:
        # Always restore original matmul
        mx.matmul = original_matmul
        return {"error": f"Benchmark failed: {str(e)}"}


def evaluate(program_path: str) -> Dict:
    """
    Main evaluation function for real LLM fine-tuning optimization
    """
    return evaluate_real_llm_finetuning(program_path)


# Stage-based evaluation for cascade evaluation
def evaluate_stage1(program_path: str) -> Dict:
    """Stage 1: Quick validation"""
    try:
        # Basic function existence check
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        required_functions = ["get_device_info", "choose_tile_size", "optimized_matmul"]
        for func_name in required_functions:
            if not hasattr(program, func_name):
                return {"valid_structure": 0.0, "error": f"Missing {func_name}"}
        
        # Quick device info test
        device_info = program.get_device_info()
        if not isinstance(device_info, dict):
            return {"valid_structure": 0.0, "error": "Invalid device_info"}
        
        # Quick tile size test
        tile_M, tile_N, tile_K = program.choose_tile_size(512, 512, 512, device_info)
        if not all(isinstance(x, int) for x in [tile_M, tile_N, tile_K]):
            return {"valid_structure": 0.0, "error": "Invalid tile sizes"}
        
        return {"valid_structure": 1.0}
        
    except Exception as e:
        return {"valid_structure": 0.0, "error": str(e)}


def evaluate_stage2(program_path: str) -> Dict:
    """Stage 2: Quick performance test with matrix operations"""
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Test with realistic LLM-sized matrices
        device_info = program.get_device_info()
        
        # Test different matrix sizes common in LLMs
        test_cases = [
            (1024, 512, 2048),  # Typical attention
            (2048, 2048, 512),  # Typical MLP
            (512, 4096, 1024),  # Embedding/output
        ]
        
        success_count = 0
        total_time = 0
        
        for M, N, K in test_cases:
            try:
                A = mx.random.normal((M, K), dtype=mx.float32)
                B = mx.random.normal((K, N), dtype=mx.float32)
                
                tile_M, tile_N, tile_K = program.choose_tile_size(M, N, K, device_info)
                
                start_time = time.perf_counter()
                if tile_M > 0 and tile_N > 0 and tile_K > 0:
                    C = program.optimized_matmul(A, B, tile_M, tile_N, tile_K)
                else:
                    C = mx.matmul(A, B)  # Direct MLX
                mx.eval(C)
                end_time = time.perf_counter()
                
                # Verify correctness
                C_ref = mx.matmul(A, B)
                error = mx.mean(mx.abs(C - C_ref))
                
                if error < 1e-3:
                    success_count += 1
                    total_time += (end_time - start_time)
                
            except Exception as e:
                print(f"Stage 2 test failed for {M}x{N}x{K}: {e}")
                continue
        
        if success_count == 0:
            return {"valid_structure": 0.0, "error": "All matrix tests failed"}
        
        # Basic performance score
        avg_time = total_time / success_count
        performance_score = min(2.0, 0.1 / avg_time)  # Normalize to reasonable range
        
        return {
            "valid_structure": 1.0,
            "performance_score": float(performance_score),
            "passes_stage2": success_count >= len(test_cases) // 2
        }
        
    except Exception as e:
        return {"valid_structure": 0.0, "error": str(e)}


def evaluate_stage3(program_path: str) -> Dict:
    """Stage 3: Full real LLM evaluation"""
    return evaluate_real_llm_finetuning(program_path)


if __name__ == "__main__":
    # Quick test of the evaluator
    print("🧪 Testing Real LLM Fine-tuning Evaluator")
    
    if not HAS_MLX_LM:
        print("❌ mlx-lm not available. Install with: pip install mlx-lm")
        exit(1)
    
    # Test with initial program
    initial_program_path = "initial_program.py"
    if os.path.exists(initial_program_path):
        print(f"Testing with {initial_program_path}...")
        results = evaluate(initial_program_path)
        print(f"Results: {results}")
    else:
        print(f"❌ {initial_program_path} not found")
