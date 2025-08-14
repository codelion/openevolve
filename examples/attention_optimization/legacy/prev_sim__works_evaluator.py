# #!/usr/bin/env python3
# """
# Evaluator for attention optimization programs.
# This script evaluates how good each evolved optimization is.
# """

# import sys
# import json
# import subprocess
# import tempfile
# import time
# import os
# from pathlib import Path

# class MLIRAttentionEvaluator:
#     """Evaluates MLIR attention optimizations"""
    
#     def __init__(self):
#         # Load base MLIR implementation
#         self.base_mlir_file = Path(__file__).parent / "mlir" / "baseline_attention.mlir"
#         self.reference_performance = None
        
#         # Test configurations (batch, heads, seq_len, head_dim)
#         self.test_configs = [
#             (1, 8, 128, 64),   # Small
#             (2, 12, 256, 64),  # Medium
#             (4, 16, 512, 64),  # Large
#         ]
    
#     def load_base_mlir(self):
#         """Load the baseline MLIR implementation"""
#         if not self.base_mlir_file.exists():
#             # Create a simple baseline if it doesn't exist
#             return self.create_baseline_mlir()
        
#         with open(self.base_mlir_file, 'r') as f:
#             return f.read()
    
#     def create_baseline_mlir(self):
#         """Create a simple baseline MLIR attention implementation"""
#         baseline = '''
#         func.func @baseline_attention(
#             %query: tensor<?x?x?x?xf32>,
#             %key: tensor<?x?x?x?xf32>, 
#             %value: tensor<?x?x?x?xf32>
#         ) -> tensor<?x?x?x?xf32> {
#             // Simple attention: Q @ K^T @ V (simplified)
#             %result = linalg.generic {
#                 indexing_maps = [affine_map<(b, h, s, d) -> (b, h, s, d)>],
#                 iterator_types = ["parallel", "parallel", "parallel", "parallel"]
#             } ins(%query : tensor<?x?x?x?xf32>) 
#               outs(%query : tensor<?x?x?x?xf32>) {
#             ^bb0(%q: f32, %out: f32):
#                 linalg.yield %q : f32
#             }
#             return %result : tensor<?x?x?x?xf32>
#         }
#         '''
#         return baseline
    
#     def compile_mlir_with_optimizations(self, base_mlir, optimization_params):
#         """Apply optimizations and compile MLIR"""
#         try:
#             # Create optimized MLIR by applying transformations
#             optimized_mlir = self.apply_optimizations(base_mlir, optimization_params)
            
#             # Simulate MLIR compilation (in real implementation, use mlir-opt)
#             compile_success = self.simulate_mlir_compilation(optimized_mlir)
            
#             return compile_success, optimized_mlir
            
#         except Exception as e:
#             return False, str(e)
    
#     def apply_optimizations(self, base_mlir, params):
#         """Apply optimization parameters to base MLIR"""
#         # In a real implementation, this would use MLIR transform dialect
#         # For now, we simulate by modifying the MLIR text
        
#         optimized = base_mlir
        
#         # Add optimization annotations as comments
#         header = f"""
#         // Optimized with parameters:
#         // Tile sizes: {params.get('tile_size_m', 32)}x{params.get('tile_size_n', 32)}x{params.get('tile_size_k', 32)}
#         // Vectorization: {params.get('vectorization', 'none')}
#         // Fusion: {params.get('fusion_strategy', 'none')}
#         // Unroll factor: {params.get('unroll_factor', 1)}
#         """
        
#         optimized = header + optimized
        
#         return optimized
    
#     def simulate_mlir_compilation(self, mlir_code):
#         """Simulate MLIR compilation success"""
#         # Simple checks for valid MLIR
#         required_elements = ['func.func', 'tensor', 'return']
        
#         for element in required_elements:
#             if element not in mlir_code:
#                 return False
        
#         # Check for obvious syntax errors
#         if mlir_code.count('{') != mlir_code.count('}'):
#             return False
            
#         return True
    
#     def benchmark_implementation(self, optimized_mlir, test_config):
#         """Benchmark the optimized implementation"""
#         batch, heads, seq_len, head_dim = test_config
        
#         # Estimate FLOPs for attention computation
#         # Q@K^T: batch * heads * seq_len^2 * head_dim
#         # Softmax@V: batch * heads * seq_len^2 * head_dim
#         flops = 2 * batch * heads * seq_len * seq_len * head_dim
        
#         # Simulate performance based on optimizations
#         base_flops_per_second = 1e12  # 1 TFLOP/s baseline
        
#         # Apply optimization factors
#         speedup_factor = self.calculate_speedup_factor(optimized_mlir)
        
#         # Calculate runtime
#         runtime = flops / (base_flops_per_second * speedup_factor)
        
#         return runtime
    
#     def calculate_speedup_factor(self, optimized_mlir):
#         """Calculate speedup factor based on applied optimizations"""
#         speedup = 1.0
        
#         # Parse optimization comments to extract speedup factors
#         if "Tile sizes: 64x64x64" in optimized_mlir:
#             speedup *= 1.15  # 15% improvement from better tiling
#         elif "Tile sizes: 32x32x32" in optimized_mlir:
#             speedup *= 1.10  # 10% improvement
        
#         if "Vectorization: full" in optimized_mlir:
#             speedup *= 1.20  # 20% improvement from vectorization
#         elif "Vectorization: outer" in optimized_mlir:
#             speedup *= 1.10  # 10% improvement
        
#         if "Fusion: producer" in optimized_mlir or "Fusion: consumer" in optimized_mlir:
#             speedup *= 1.08  # 8% improvement from fusion
#         elif "Fusion: both" in optimized_mlir:
#             speedup *= 1.15  # 15% improvement
        
#         if "Unroll factor: 4" in optimized_mlir:
#             speedup *= 1.05  # 5% improvement from unrolling
#         elif "Unroll factor: 8" in optimized_mlir:
#             speedup *= 1.08  # 8% improvement
        
#         return speedup
    
#     def get_reference_performance(self):
#         """Get baseline performance for comparison"""
#         if self.reference_performance is None:
#             base_mlir = self.load_base_mlir()
#             total_time = 0
            
#             for config in self.test_configs:
#                 runtime = self.benchmark_implementation(base_mlir, config)
#                 total_time += runtime
            
#             self.reference_performance = total_time / len(self.test_configs)
        
#         return self.reference_performance

#     def evaluate_program(self, program_content):
#         """Main evaluation function called by OpenEvolve"""
#         try:
#             # Execute the evolved program to get optimization parameters
#             exec_globals = {}
#             exec(program_content, exec_globals)
            
#             if 'optimize_attention' not in exec_globals:
#                 return {
#                     "error": "No optimize_attention function found",
#                     "score": 0.0
#                 }
            
#             # Get optimization parameters
#             params = exec_globals['optimize_attention']()
            
#             # Load base MLIR
#             base_mlir = self.load_base_mlir()
            
#             # Apply optimizations and compile
#             success, optimized_mlir = self.compile_mlir_with_optimizations(base_mlir, params)
            
#             if not success:
#                 return {
#                     "error": f"Compilation failed: {optimized_mlir}",
#                     "score": 0.0
#                 }
            
#             # Benchmark performance
#             total_runtime = 0
#             for config in self.test_configs:
#                 runtime = self.benchmark_implementation(optimized_mlir, config)
#                 total_runtime += runtime
            
#             avg_runtime = total_runtime / len(self.test_configs)
            
#             # Calculate speedup vs reference
#             reference_time = self.get_reference_performance()
#             speedup = reference_time / avg_runtime if avg_runtime > 0 else 0.0
            
#             # Score is the speedup (higher is better)
#             score = speedup
            
#             return {
#                 "score": score,
#                 "speedup": speedup,
#                 "runtime": avg_runtime,
#                 "reference_runtime": reference_time,
#                 "optimizations": params,
#                 "success": True
#             }
            
#         except Exception as e:
#             return {
#                 "error": str(e),
#                 "score": 0.0
#             }

# def evaluate(program_path):
#     try:
#         with open(program_path, 'r') as f:
#             program_content = f.read()
        
#         evaluator = MLIRAttentionEvaluator()
#         result = evaluator.evaluate_program(program_content)
        
#         print(json.dumps(result))
        
#     except Exception as e:
#         error_result = {
#             "error": str(e),
#             "score": 0.0
#         }
#         print(json.dumps(error_result))



#!/usr/bin/env python3
"""
Fixed evaluator for attention optimization programs.
This script evaluates how good each evolved optimization is.
"""

import sys
import json
import subprocess
import tempfile
import time
import os
from pathlib import Path

class MLIRAttentionEvaluator:
    """Evaluates MLIR attention optimizations"""
    
    def __init__(self):
        # Load base MLIR implementation
        self.base_mlir_file = Path(__file__).parent / "mlir" / "baseline_attention.mlir"
        self.reference_performance = None
        
        # Test configurations (batch, heads, seq_len, head_dim)
        self.test_configs = [
            (1, 8, 128, 64),   # Small
            (2, 12, 256, 64),  # Medium
            (4, 16, 512, 64),  # Large
        ]
    
    def load_base_mlir(self):
        """Load the baseline MLIR implementation"""
        if not self.base_mlir_file.exists():
            # Create a simple baseline if it doesn't exist
            return self.create_baseline_mlir()
        
        with open(self.base_mlir_file, 'r') as f:
            return f.read()
    
    def create_baseline_mlir(self):
        """Create a simple baseline MLIR attention implementation"""
        baseline = '''
        func.func @baseline_attention(
            %query: tensor<?x?x?x?xf32>,
            %key: tensor<?x?x?x?xf32>, 
            %value: tensor<?x?x?x?xf32>
        ) -> tensor<?x?x?x?xf32> {
            // Simple attention: Q @ K^T @ V (simplified)
            %result = linalg.generic {
                indexing_maps = [affine_map<(b, h, s, d) -> (b, h, s, d)>],
                iterator_types = ["parallel", "parallel", "parallel", "parallel"]
            } ins(%query : tensor<?x?x?x?xf32>) 
              outs(%query : tensor<?x?x?x?xf32>) {
            ^bb0(%q: f32, %out: f32):
                linalg.yield %q : f32
            }
            return %result : tensor<?x?x?x?xf32>
        }
        '''
        return baseline
    
    def compile_mlir_with_optimizations(self, base_mlir, optimization_params):
        """Apply optimizations and compile MLIR"""
        try:
            # Create optimized MLIR by applying transformations
            optimized_mlir = self.apply_optimizations(base_mlir, optimization_params)
            
            # Simulate MLIR compilation (in real implementation, use mlir-opt)
            compile_success = self.simulate_mlir_compilation(optimized_mlir)
            
            return compile_success, optimized_mlir
            
        except Exception as e:
            return False, str(e)
    
    def apply_optimizations(self, base_mlir, params):
        """Apply optimization parameters to base MLIR"""
        # In a real implementation, this would use MLIR transform dialect
        # For now, we simulate by modifying the MLIR text
        
        optimized = base_mlir
        
        # Add optimization annotations as comments
        header = f"""
        // Optimized with parameters:
        // Tile sizes: {params.get('tile_size_m', 32)}x{params.get('tile_size_n', 32)}x{params.get('tile_size_k', 32)}
        // Vectorization: {params.get('vectorization', 'none')}
        // Fusion: {params.get('fusion_strategy', 'none')}
        // Unroll factor: {params.get('unroll_factor', 1)}
        """
        
        optimized = header + optimized
        
        return optimized
    
    def simulate_mlir_compilation(self, mlir_code):
        """Simulate MLIR compilation success"""
        # Simple checks for valid MLIR
        required_elements = ['func.func', 'tensor', 'return']
        
        for element in required_elements:
            if element not in mlir_code:
                return False
        
        # Check for obvious syntax errors
        if mlir_code.count('{') != mlir_code.count('}'):
            return False
            
        return True
    
    def benchmark_implementation(self, optimized_mlir, test_config):
        """Benchmark the optimized implementation"""
        batch, heads, seq_len, head_dim = test_config
        
        # Estimate FLOPs for attention computation
        # Q@K^T: batch * heads * seq_len^2 * head_dim
        # Softmax@V: batch * heads * seq_len^2 * head_dim
        flops = 2 * batch * heads * seq_len * seq_len * head_dim
        
        # Simulate performance based on optimizations
        base_flops_per_second = 1e12  # 1 TFLOP/s baseline
        
        # Apply optimization factors
        speedup_factor = self.calculate_speedup_factor(optimized_mlir)
        
        # Calculate runtime
        runtime = flops / (base_flops_per_second * speedup_factor)
        
        return runtime
    
    def calculate_speedup_factor(self, optimized_mlir):
        """Calculate speedup factor based on applied optimizations"""
        speedup = 1.0
        
        # Parse optimization comments to extract speedup factors
        if "Tile sizes: 128x128x128" in optimized_mlir:
            speedup *= 1.25  # 25% improvement from large tiles
        elif "Tile sizes: 64x64x64" in optimized_mlir:
            speedup *= 1.15  # 15% improvement from better tiling
        elif "Tile sizes: 32x32x32" in optimized_mlir:
            speedup *= 1.10  # 10% improvement
        elif "Tile sizes: 256x256x256" in optimized_mlir:
            speedup *= 1.30  # 30% improvement from very large tiles
        
        if "Vectorization: full" in optimized_mlir:
            speedup *= 1.20  # 20% improvement from vectorization
        elif "Vectorization: outer" in optimized_mlir:
            speedup *= 1.10  # 10% improvement
        elif "Vectorization: inner" in optimized_mlir:
            speedup *= 1.08  # 8% improvement
        
        if "Fusion: producer" in optimized_mlir or "Fusion: consumer" in optimized_mlir:
            speedup *= 1.08  # 8% improvement from fusion
        elif "Fusion: both" in optimized_mlir:
            speedup *= 1.15  # 15% improvement
        
        if "Unroll factor: 8" in optimized_mlir:
            speedup *= 1.08  # 8% improvement
        elif "Unroll factor: 4" in optimized_mlir:
            speedup *= 1.05  # 5% improvement from unrolling
        elif "Unroll factor: 2" in optimized_mlir:
            speedup *= 1.02  # 2% improvement
        
        return speedup
    
    def get_reference_performance(self):
        """Get baseline performance for comparison"""
        if self.reference_performance is None:
            base_mlir = self.load_base_mlir()
            total_time = 0
            
            for config in self.test_configs:
                runtime = self.benchmark_implementation(base_mlir, config)
                total_time += runtime
            
            self.reference_performance = total_time / len(self.test_configs)
        
        return self.reference_performance


def evaluate(program_path):
    """
    Main evaluation function called by OpenEvolve.
    
    IMPORTANT: OpenEvolve expects this exact function signature!
    It should return a dictionary with metrics.
    """
    try:
        # Execute the evolved program to get optimization parameters
        with open(program_path, 'r') as f:
            program_content = f.read()

        exec_globals = {}
        exec(program_content, exec_globals)
        
        if 'optimize_attention' not in exec_globals:
            # Return error metric (higher error = worse performance)
            return {"error": 1000.0}
        
        # Get optimization parameters
        params = exec_globals['optimize_attention']()
        
        # Global evaluator instance
        evaluator = MLIRAttentionEvaluator()

        # Load base MLIR
        base_mlir = evaluator.load_base_mlir()
        
        # Apply optimizations and compile
        success, optimized_mlir = evaluator.compile_mlir_with_optimizations(base_mlir, params)
        
        if not success:
            # Return high error for compilation failure
            return {"error": 500.0}
        
        # Benchmark performance
        total_runtime = 0
        for config in evaluator.test_configs:
            runtime = evaluator.benchmark_implementation(optimized_mlir, config)
            total_runtime += runtime
        
        avg_runtime = total_runtime / len(evaluator.test_configs)
        
        # Calculate speedup vs reference
        reference_time = evaluator.get_reference_performance()
        speedup = reference_time / avg_runtime if avg_runtime > 0 else 0.0
        
        # Convert speedup to error metric (lower error = better performance)
        # Target is 1.32x speedup (32% improvement like AlphaEvolve)
        target_speedup = 1.32
        
        if speedup >= target_speedup:
            # Achieved target! Very low error
            error = max(0.1, (target_speedup - speedup) * 10)
        else:
            # Below target, error increases as speedup decreases
            error = (target_speedup - speedup) * 100
        
        # Ensure error is positive (OpenEvolve minimizes error)
        error = max(0.01, error)
        
        # Return metrics in OpenEvolve format
        result = {
            "error": error,
            "speedup": speedup,
            "runtime": avg_runtime,
            "reference_runtime": reference_time,
        }
        
        # Add debug info as additional metrics
        for key, value in params.items():
            if isinstance(value, (int, float, bool)):
                result[f"param_{key}"] = float(value) if isinstance(value, bool) else value
        
        return result
        
    except Exception as e:
        # Return very high error for any exception
        return {"error": 1000.0, "exception": str(e)}