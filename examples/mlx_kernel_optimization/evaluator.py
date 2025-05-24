"""
Evaluator for MLX-LM performance optimization
Tests real inference and training performance with Qwen2.5-0.5B-Instruct-bf16
"""

import importlib.util
import time
import traceback
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import tempfile
import os
import gc


def evaluate(program_path):
    """
    Evaluate MLX-LM optimization by measuring real inference and training performance
    
    Args:
        program_path: Path to the program file
        
    Returns:
        Dictionary of performance metrics
    """
    
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Check required functions exist
        required_functions = ["get_device_info", "choose_tile_size", "optimized_matmul"]
        for func_name in required_functions:
            if not hasattr(program, func_name):
                return {
                    "inference_speedup": 0.0,
                    "training_speedup": 0.0,
                    "combined_score": 0.0,
                    "error": f"Missing {func_name} function"
                }
        
        # Test MLX-LM optimization
        inference_results = test_mlx_lm_inference(program)
        training_results = test_mlx_lm_training(program)
        
        # Calculate combined score
        inference_speedup = inference_results.get("speedup", 0.0)
        training_speedup = training_results.get("speedup", 0.0)
        
        # Weighted scoring: 60% inference, 40% training (inference is more common)
        combined_score = 0.6 * inference_speedup + 0.4 * training_speedup
        
        # Bonus for consistency (both working well)
        if inference_speedup > 1.02 and training_speedup > 1.02:
            combined_score *= 1.1  # 10% bonus for consistent optimization
        
        return {
            "inference_speedup": float(inference_speedup),
            "training_speedup": float(training_speedup),
            "inference_time_original": float(inference_results.get("original_time", 0.0)),
            "inference_time_optimized": float(inference_results.get("optimized_time", 0.0)),
            "training_time_original": float(training_results.get("original_time", 0.0)),
            "training_time_optimized": float(training_results.get("optimized_time", 0.0)),
            "combined_score": float(combined_score),
            "peak_memory_mb": float(inference_results.get("peak_memory_mb", 0.0)),
            "model_loaded": bool(inference_results.get("model_loaded", False)),
            "error_inference": inference_results.get("error", ""),
            "error_training": training_results.get("error", "")
        }
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        traceback.print_exc()
        return {
            "inference_speedup": 0.0,
            "training_speedup": 0.0,
            "combined_score": 0.0,
            "error": str(e)
        }


def test_mlx_lm_inference(program):
    """Test MLX-LM inference performance with optimization"""
    
    try:
        # Import MLX-LM
        try:
            from mlx_lm import load, generate
        except ImportError:
            return {"speedup": 0.0, "error": "mlx-lm not installed"}
        
        # Store original matmul
        original_matmul = mx.matmul
        
        # Get device info
        device_info = program.get_device_info()
        
        # Create optimized matmul function
        def create_optimized_matmul():
            def optimized_matmul(A, B):
                # Only optimize 2D matrices above threshold
                if (len(A.shape) == 2 and len(B.shape) == 2 and 
                    A.shape[0] * A.shape[1] * B.shape[1] > 50_000):  # Lower threshold for inference
                    
                    M, K1 = A.shape
                    K2, N = B.shape
                    
                    if K1 == K2:
                        tile_M, tile_N, tile_K = program.choose_tile_size(M, N, K1, device_info)
                        return program.optimized_matmul(A, B, tile_M, tile_N, tile_K)
                
                return original_matmul(A, B)
            return optimized_matmul
        
        # Load model (small model for fast testing)
        model_name = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"
        
        try:
            model, tokenizer = load(model_name)
        except Exception as e:
            # Fallback to any available small model
            try:
                model, tokenizer = load("mlx-community/SmolLM-135M")
            except:
                return {"speedup": 0.0, "error": f"Could not load model: {str(e)}"}
        
        # Test prompts
        test_prompts = [
            "Hello, how are you?",
            "What is machine learning?",
            "Explain Python programming",
            "Tell me about Apple Silicon"
        ]
        
        # Test with original MLX
        mx.matmul = original_matmul
        
        # Warmup
        for _ in range(2):
            try:
                _ = generate(model, tokenizer, prompt="Hello", max_tokens=10, verbose=False)
            except:
                pass
        
        # Benchmark original
        original_times = []
        for prompt in test_prompts:
            start_time = time.perf_counter()
            try:
                response = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
                mx.eval(response)
            except Exception as e:
                print(f"Generation failed: {e}")
                continue
            end_time = time.perf_counter()
            original_times.append(end_time - start_time)
        
        if not original_times:
            return {"speedup": 0.0, "error": "Could not generate text"}
        
        original_time = np.median(original_times)
        
        # Test with optimized MLX
        optimized_matmul_func = create_optimized_matmul()
        mx.matmul = optimized_matmul_func
        
        # Warmup
        for _ in range(2):
            try:
                _ = generate(model, tokenizer, prompt="Hello", max_tokens=10, verbose=False)
            except:
                pass
        
        # Benchmark optimized
        optimized_times = []
        for prompt in test_prompts:
            start_time = time.perf_counter()
            try:
                response = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
                mx.eval(response)
            except Exception as e:
                print(f"Optimized generation failed: {e}")
                continue
            end_time = time.perf_counter()
            optimized_times.append(end_time - start_time)
        
        # Restore original
        mx.matmul = original_matmul
        
        if not optimized_times:
            return {"speedup": 0.0, "error": "Optimized generation failed"}
        
        optimized_time = np.median(optimized_times)
        speedup = original_time / optimized_time if optimized_time > 0 else 0.0
        
        # Clean up
        del model, tokenizer
        gc.collect()
        
        return {
            "speedup": speedup,
            "original_time": original_time,
            "optimized_time": optimized_time,
            "model_loaded": True,
            "peak_memory_mb": 0.0  # Could add memory monitoring here
        }
        
    except Exception as e:
        # Always restore original matmul
        mx.matmul = original_matmul
        return {"speedup": 0.0, "error": str(e)}


def test_mlx_lm_training(program):
    """Test training performance with optimization"""
    
    try:
        # Store original matmul
        original_matmul = mx.matmul
        
        # Create a minimal training scenario
        class SimpleLanguageModel(nn.Module):
            def __init__(self, vocab_size=1000, hidden_dim=256, seq_len=128):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_dim)
                self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
                self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
                self.output = nn.Linear(hidden_dim, vocab_size)
                
            def __call__(self, x):
                x = self.embedding(x)
                x = nn.gelu(self.linear1(x))
                x = self.linear2(x)
                return self.output(x)
        
        # Training configuration
        batch_size = 8
        seq_len = 128
        vocab_size = 1000
        hidden_dim = 256
        
        # Get device info
        device_info = program.get_device_info()
        
        # Create optimized matmul function
        def create_optimized_matmul():
            def optimized_matmul(A, B):
                # Training uses larger matrices, so higher threshold
                if (len(A.shape) == 2 and len(B.shape) == 2 and 
                    A.shape[0] * A.shape[1] * B.shape[1] > 100_000):
                    
                    M, K1 = A.shape
                    K2, N = B.shape
                    
                    if K1 == K2:
                        tile_M, tile_N, tile_K = program.choose_tile_size(M, N, K1, device_info)
                        return program.optimized_matmul(A, B, tile_M, tile_N, tile_K)
                
                return original_matmul(A, B)
            return optimized_matmul
        
        # Create model and data
        model = SimpleLanguageModel(vocab_size, hidden_dim, seq_len)
        optimizer = optim.Adam(learning_rate=1e-3)
        
        # Training function
        def training_step():
            # Generate random batch
            inputs = mx.random.randint(0, vocab_size, (batch_size, seq_len))
            targets = mx.random.randint(0, vocab_size, (batch_size, seq_len))
            
            def loss_fn(model, inputs, targets):
                logits = model(inputs)
                return nn.losses.cross_entropy(
                    logits.reshape(-1, vocab_size), 
                    targets.reshape(-1), 
                    reduction='mean'
                )
            
            # Forward and backward pass
            loss, grads = mx.value_and_grad(loss_fn)(model, inputs, targets)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            
            return loss
        
        # Test with original MLX
        mx.matmul = original_matmul
        
        # Warmup
        for _ in range(3):
            training_step()
        
        # Benchmark original
        original_times = []
        for _ in range(5):
            start_time = time.perf_counter()
            training_step()
            end_time = time.perf_counter()
            original_times.append(end_time - start_time)
        
        original_time = np.median(original_times)
        
        # Test with optimized MLX
        optimized_matmul_func = create_optimized_matmul()
        mx.matmul = optimized_matmul_func
        
        # Warmup
        for _ in range(3):
            training_step()
        
        # Benchmark optimized
        optimized_times = []
        for _ in range(5):
            start_time = time.perf_counter()
            training_step()
            end_time = time.perf_counter()
            optimized_times.append(end_time - start_time)
        
        # Restore original
        mx.matmul = original_matmul
        
        optimized_time = np.median(optimized_times)
        speedup = original_time / optimized_time if optimized_time > 0 else 0.0
        
        # Clean up
        del model, optimizer
        gc.collect()
        
        return {
            "speedup": speedup,
            "original_time": original_time,
            "optimized_time": optimized_time
        }
        
    except Exception as e:
        # Always restore original matmul
        mx.matmul = original_matmul
        return {"speedup": 0.0, "error": str(e)}


# Stage-based evaluation for cascade evaluation
def evaluate_stage1(program_path):
    """First stage - quick validation"""
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Check required functions
        required = ["get_device_info", "choose_tile_size", "optimized_matmul"]
        for func_name in required:
            if not hasattr(program, func_name):
                return {"valid_structure": 0.0, "error": f"Missing {func_name}"}
        
        # Quick test
        device_info = program.get_device_info()
        tile_M, tile_N, tile_K = program.choose_tile_size(256, 256, 256, device_info)
        
        if not (1 <= tile_M <= 256 and 1 <= tile_N <= 256 and 1 <= tile_K <= 256):
            return {"valid_structure": 0.5, "error": "Invalid tile sizes"}
        
        return {"valid_structure": 1.0}
        
    except Exception as e:
        return {"valid_structure": 0.0, "error": str(e)}


def evaluate_stage2(program_path):
    """Second stage - quick performance test"""
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Quick matrix multiplication test
        A = mx.random.normal((128, 256))
        B = mx.random.normal((256, 128))
        
        device_info = program.get_device_info()
        tile_M, tile_N, tile_K = program.choose_tile_size(128, 128, 256, device_info)
        
        # Test optimized matmul function
        start_time = time.perf_counter()
        C = program.optimized_matmul(A, B, tile_M, tile_N, tile_K)
        mx.eval(C)
        elapsed = time.perf_counter() - start_time
        
        # Verify correctness
        C_ref = mx.matmul(A, B)
        error = mx.mean(mx.abs(C - C_ref))
        
        if error > 1e-3:
            return {"valid_structure": 0.0, "error": "Incorrect computation"}
        
        quick_score = min(1.0, 0.1 / elapsed)  # Faster = better score
        
        return {
            "valid_structure": 1.0,
            "quick_score": float(quick_score),
            "passes_stage2": quick_score > 0.5
        }
        
    except Exception as e:
        return {"valid_structure": 0.0, "error": str(e)}


def evaluate_stage3(program_path):
    """Third stage - full MLX-LM evaluation"""
    return evaluate(program_path)
