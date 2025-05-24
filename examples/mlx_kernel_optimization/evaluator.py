"""
Evaluator for MLX Training Performance Optimization (Training-Only Focus)
"""

import importlib.util
import time
import traceback
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import gc
import sys
import io


def evaluate(program_path):
    """
    Evaluate MLX training optimization (training-only focus)
    
    Args:
        program_path: Path to the program file
        
    Returns:
        Dictionary of performance metrics
    """
    
    try:
        # Load the program with better error handling
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        
        # Capture any import/execution errors
        try:
            spec.loader.exec_module(program)
        except Exception as load_error:
            return {
                "training_speedup": 0.0,
                "combined_score": 0.0,
                "error": f"Failed to load program: {str(load_error)}"
            }
        
        # Check required functions exist
        required_functions = ["get_device_info", "choose_tile_size", "optimized_matmul"]
        missing_functions = []
        for func_name in required_functions:
            if not hasattr(program, func_name):
                missing_functions.append(func_name)
        
        if missing_functions:
            return {
                "training_speedup": 0.0,
                "combined_score": 0.0,
                "error": f"Missing functions: {', '.join(missing_functions)}"
            }
        
        # Test training optimization with enhanced evaluation
        training_results = test_training_performance_enhanced(program)
        
        # Calculate combined score (training-only)
        training_speedup = training_results.get("speedup", 0.0)
        
        # Simple scoring: training speedup with bonuses for good performance
        combined_score = training_speedup
        
        # Bonus multipliers for significant improvements
        if training_speedup > 1.20:  # >20% improvement
            combined_score *= 1.4
        elif training_speedup > 1.15:  # >15% improvement
            combined_score *= 1.3
        elif training_speedup > 1.10:  # >10% improvement
            combined_score *= 1.2
        elif training_speedup > 1.05:  # >5% improvement
            combined_score *= 1.1
        
        return {
            "training_speedup": float(training_speedup),
            "training_time_original": float(training_results.get("original_time", 0.0)),
            "training_time_optimized": float(training_results.get("optimized_time", 0.0)),
            "combined_score": float(combined_score),
            "optimizations_applied": int(training_results.get("optimizations_applied", 0)),
            "matrix_operations_count": int(training_results.get("matrix_ops", 0)),
            "test_successful": bool(training_results.get("test_successful", False)),
            "debug_info": {
                "training_error": training_results.get("error", ""),
                "matrix_sizes_tested": training_results.get("matrix_sizes", []),
                "device_info": training_results.get("device_info", {})
            }
        }
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        traceback.print_exc()
        return {
            "training_speedup": 0.0,
            "combined_score": 0.0,
            "error": f"Evaluation exception: {str(e)}"
        }


def test_training_performance_enhanced(program):
    """Test MLX training performance with enhanced setup and debugging"""
    
    try:
        # Store original matmul
        original_matmul = mx.matmul
        
        # Get device info with error handling
        try:
            device_info = program.get_device_info()
        except Exception as e:
            return {"speedup": 0.0, "error": f"get_device_info failed: {str(e)}", "test_successful": False}
        
        # Track optimizations applied
        optimization_count = 0
        matrix_sizes = []
        
        # Test basic function calls first
        try:
            # Test choose_tile_size with simple inputs
            tile_M, tile_N, tile_K = program.choose_tile_size(256, 256, 256, device_info)
            if not (isinstance(tile_M, int) and isinstance(tile_N, int) and isinstance(tile_K, int)):
                return {"speedup": 0.0, "error": "choose_tile_size returned non-integer values", "test_successful": False}
            if not (1 <= tile_M <= 256 and 1 <= tile_N <= 256 and 1 <= tile_K <= 256):
                return {"speedup": 0.0, "error": f"choose_tile_size returned invalid sizes: {tile_M}, {tile_N}, {tile_K}", "test_successful": False}
        except Exception as e:
            return {"speedup": 0.0, "error": f"choose_tile_size failed: {str(e)}", "test_successful": False}
        
        # Test optimized_matmul with simple matrices
        try:
            A_test = mx.random.normal((64, 64), dtype=mx.float32)
            B_test = mx.random.normal((64, 64), dtype=mx.float32)
            C_test = program.optimized_matmul(A_test, B_test, tile_M, tile_N, tile_K)
            mx.eval(C_test)  # Force evaluation
            
            # Verify correctness
            C_ref = mx.matmul(A_test, B_test)
            error = mx.mean(mx.abs(C_test - C_ref))
            if error > 1e-3:
                return {"speedup": 0.0, "error": f"optimized_matmul produces incorrect results, error: {float(error)}", "test_successful": False}
        except Exception as e:
            return {"speedup": 0.0, "error": f"optimized_matmul failed: {str(e)}", "test_successful": False}
        
        # Create optimized matmul with debugging and lower threshold
        def create_optimized_matmul():
            def optimized_matmul_debug(A, B):
                nonlocal optimization_count, matrix_sizes
                
                # Lower threshold for training - catch more operations
                if (len(A.shape) == 2 and len(B.shape) == 2 and 
                    A.shape[0] * A.shape[1] * B.shape[1] > 15_000):  # Lower threshold
                    
                    M, K1 = A.shape
                    K2, N = B.shape
                    
                    if K1 == K2:
                        matrix_sizes.append((M, K1, N, M * K1 * N))
                        optimization_count += 1
                        
                        try:
                            tile_M, tile_N, tile_K = program.choose_tile_size(M, N, K1, device_info)
                            return program.optimized_matmul(A, B, tile_M, tile_N, tile_K)
                        except Exception as opt_error:
                            # Fall back to original if optimization fails
                            print(f"Optimization failed for {M}x{K1}x{N}: {opt_error}")
                            return original_matmul(A, B)
                
                return original_matmul(A, B)
            return optimized_matmul_debug
        
        # Create enhanced training model - larger and more matrix-heavy
        class EnhancedTrainingModel(nn.Module):
            def __init__(self, vocab_size=4000, hidden_dim=768, seq_len=256):  # Smaller for stability
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_dim)
                
                # Multiple transformer-like layers with heavy matrix operations
                self.layers = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 3),  # MLP expansion  
                    nn.GELU(),
                    nn.Linear(hidden_dim * 3, hidden_dim),  # MLP projection
                    nn.Linear(hidden_dim, hidden_dim),      # Residual connection
                    nn.Linear(hidden_dim, hidden_dim * 2),  # Another expansion
                    nn.GELU(), 
                    nn.Linear(hidden_dim * 2, hidden_dim),  # Another projection
                )
                
                # Attention-like operations
                self.attention_layers = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),      # Query projection
                    nn.Linear(hidden_dim, hidden_dim),      # Key projection  
                    nn.Linear(hidden_dim, hidden_dim),      # Value projection
                    nn.Linear(hidden_dim, hidden_dim),      # Output projection
                )
                
                self.output = nn.Linear(hidden_dim, vocab_size)  # Large output
                
            def __call__(self, x):
                x = self.embedding(x)  # [batch, seq, hidden]
                
                # Apply multiple linear transformations
                x = self.layers(x)
                x = self.attention_layers(x)
                
                return self.output(x)
        
        # Enhanced training configuration for more matrix operations but stable
        batch_size = 16      # Moderate batch size for stability
        seq_len = 256        # Moderate sequence length
        vocab_size = 4000    # Moderate vocabulary  
        hidden_dim = 768     # Moderate hidden dimension
        
        # Create model and optimizer
        model = EnhancedTrainingModel(vocab_size, hidden_dim, seq_len)
        optimizer = optim.Adam(learning_rate=1e-3)
        
        # Training function with forward + backward passes
        def enhanced_training_step():
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
            
            # Forward and backward pass (this is where the matrix ops happen)
            loss, grads = mx.value_and_grad(loss_fn)(model, inputs, targets)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            
            return loss
        
        # Test with original MLX
        mx.matmul = original_matmul
        
        # Extended warmup for stable timing
        for _ in range(5):
            enhanced_training_step()
        
        # Benchmark original MLX with more iterations
        original_times = []
        for _ in range(12):  # Moderate number for stability
            start_time = time.perf_counter()
            enhanced_training_step()
            end_time = time.perf_counter()
            original_times.append(end_time - start_time)
        
        # Remove outliers (top and bottom 10%)
        original_times = sorted(original_times)[1:-1]
        original_time = np.median(original_times)
        
        # Test with optimized MLX
        optimized_matmul_func = create_optimized_matmul()
        mx.matmul = optimized_matmul_func
        
        # Reset counters
        optimization_count = 0
        matrix_sizes = []
        
        # Extended warmup for optimized version
        for _ in range(5):
            enhanced_training_step()
        
        # Benchmark optimized MLX
        optimized_times = []
        for _ in range(12):  # Moderate number for stability
            start_time = time.perf_counter()
            enhanced_training_step()
            end_time = time.perf_counter()
            optimized_times.append(end_time - start_time)
        
        # Restore original
        mx.matmul = original_matmul
        
        # Remove outliers
        optimized_times = sorted(optimized_times)[1:-1]
        optimized_time = np.median(optimized_times)
        
        speedup = original_time / optimized_time if optimized_time > 0 else 0.0
        
        # Clean up
        del model, optimizer
        gc.collect()
        
        print(f"   🔧 Matrix optimizations applied: {optimization_count}")
        print(f"   📊 Unique matrix patterns: {len(set(matrix_sizes))}")
        if matrix_sizes:
            largest = max(matrix_sizes, key=lambda x: x[3])
            print(f"   📏 Largest matrix: {largest[0]}×{largest[1]}×{largest[2]} ({largest[3]:,} elements)")
        
        return {
            "speedup": speedup,
            "original_time": original_time,
            "optimized_time": optimized_time,
            "test_successful": True,
            "optimizations_applied": optimization_count,
            "matrix_sizes": matrix_sizes,
            "matrix_ops": len(matrix_sizes),
            "device_info": device_info
        }
        
    except Exception as e:
        # Always restore original matmul
        mx.matmul = original_matmul
        return {"speedup": 0.0, "error": f"Training test failed: {str(e)}", "test_successful": False}


# Stage-based evaluation for cascade evaluation with better error reporting
def evaluate_stage1(program_path):
    """First stage - quick validation with detailed error reporting"""
    try:
        # Read the program file first to check for basic structure
        with open(program_path, 'r') as f:
            program_code = f.read()
        
        # Check if the code has the required structure
        required_functions = ["get_device_info", "choose_tile_size", "optimized_matmul"]
        missing_functions = []
        for func_name in required_functions:
            if f"def {func_name}(" not in program_code:
                missing_functions.append(func_name)
        
        if missing_functions:
            return {"valid_structure": 0.0, "error": f"Missing function definitions: {', '.join(missing_functions)}"}
        
        # Try to load and execute the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        
        try:
            spec.loader.exec_module(program)
        except Exception as load_error:
            return {"valid_structure": 0.0, "error": f"Failed to load program: {str(load_error)}"}
        
        # Check required functions are actually available
        for func_name in required_functions:
            if not hasattr(program, func_name):
                return {"valid_structure": 0.0, "error": f"Function {func_name} not found after loading"}
        
        # Quick functional test
        try:
            device_info = program.get_device_info()
            tile_M, tile_N, tile_K = program.choose_tile_size(512, 512, 512, device_info)
            
            # Validate tile sizes
            if not (isinstance(tile_M, int) and isinstance(tile_N, int) and isinstance(tile_K, int)):
                return {"valid_structure": 0.0, "error": f"choose_tile_size returned non-integers: {type(tile_M)}, {type(tile_N)}, {type(tile_K)}"}
            
            if not (1 <= tile_M <= 512 and 1 <= tile_N <= 512 and 1 <= tile_K <= 512):
                return {"valid_structure": 0.5, "error": f"Invalid tile sizes: {tile_M}, {tile_N}, {tile_K}"}
            
            # Test optimized_matmul with small matrices
            A = mx.random.normal((32, 32), dtype=mx.float32)
            B = mx.random.normal((32, 32), dtype=mx.float32)
            C = program.optimized_matmul(A, B, 32, 32, 32)
            mx.eval(C)  # Force evaluation
            
        except Exception as test_error:
            return {"valid_structure": 0.0, "error": f"Function test failed: {str(test_error)}"}
        
        return {"valid_structure": 1.0}
        
    except Exception as e:
        return {"valid_structure": 0.0, "error": f"Stage 1 evaluation failed: {str(e)}"}


def evaluate_stage2(program_path):
    """Second stage - quick performance test"""
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Test with training-sized matrices
        A = mx.random.normal((128, 512), dtype=mx.float32)
        B = mx.random.normal((512, 256), dtype=mx.float32)
        
        device_info = program.get_device_info()
        tile_M, tile_N, tile_K = program.choose_tile_size(128, 256, 512, device_info)
        
        # Test optimized matmul function
        start_time = time.perf_counter()
        C = program.optimized_matmul(A, B, tile_M, tile_N, tile_K)
        mx.eval(C)
        elapsed = time.perf_counter() - start_time
        
        # Verify correctness
        C_ref = mx.matmul(A, B)
        error = mx.mean(mx.abs(C - C_ref))
        
        if error > 1e-3:
            return {"valid_structure": 0.0, "error": f"Incorrect computation, error: {float(error)}"}
        
        quick_score = min(3.0, 0.05 / elapsed)  # Generous scoring for stage 2
        
        return {
            "valid_structure": 1.0,
            "quick_score": float(quick_score),
            "passes_stage2": quick_score > 0.3  # Lower threshold
        }
        
    except Exception as e:
        return {"valid_structure": 0.0, "error": f"Stage 2 failed: {str(e)}"}


def evaluate_stage3(program_path):
    """Third stage - full training evaluation"""
    return evaluate(program_path)
