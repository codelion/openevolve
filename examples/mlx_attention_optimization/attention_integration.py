"""
MLX Attention Integration Helper

This module provides utilities to easily integrate OpenEvolve-optimized attention
into existing MLX models for side-by-side comparison and deployment.

Key features:
- Load any MLX model with optimized attention
- Compare standard vs optimized attention performance  
- Minimal code changes required (2-3 lines)
- Support for popular models (Qwen, Llama, etc.)
"""

import importlib.util
import os
import time
from typing import Dict, Optional, Tuple, Any

import mlx.core as mx
import mlx.nn as nn

try:
    import mlx_lm
    from mlx_lm import load, generate
    MLX_LM_AVAILABLE = True
except ImportError:
    print("⚠️  mlx_lm not available. Real model integration will be limited.")
    MLX_LM_AVAILABLE = False


class OptimizedAttentionWrapper:
    """Wrapper to replace standard attention with optimized version"""
    
    def __init__(self, evolved_program_path: str):
        """
        Initialize with path to evolved attention program
        
        Args:
            evolved_program_path: Path to the best_program.py from OpenEvolve
        """
        self.evolved_program_path = evolved_program_path
        self.evolved_module = None
        self._load_evolved_module()
    
    def _load_evolved_module(self):
        """Load the evolved attention module"""
        if not os.path.exists(self.evolved_program_path):
            raise FileNotFoundError(f"Evolved program not found: {self.evolved_program_path}")
        
        spec = importlib.util.spec_from_file_location("evolved_attention", self.evolved_program_path)
        self.evolved_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.evolved_module)
        
        if not hasattr(self.evolved_module, 'create_test_attention_module'):
            raise AttributeError("Evolved module missing create_test_attention_module function")
    
    def create_optimized_attention(self, hidden_size: int, num_heads: int, 
                                 num_kv_heads: int = None, **kwargs) -> nn.Module:
        """
        Create optimized attention module compatible with model architecture
        
        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            num_kv_heads: Number of key-value heads (for GQA)
            **kwargs: Additional parameters (window_size, query_chunk_size, etc.)
            
        Returns:
            Optimized attention module
        """
        if num_kv_heads is None:
            num_kv_heads = num_heads
        
        head_dim = hidden_size // num_heads
        
        # Set reasonable defaults for optimization parameters
        default_kwargs = {
            'window_size': 64,          # Enable windowed attention
            'query_chunk_size': 64,     # Enable chunking  
            'dilation_rate': 1          # No dilation by default
        }
        default_kwargs.update(kwargs)
        
        try:
            return self.evolved_module.create_test_attention_module(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                **default_kwargs
            )
        except TypeError:
            # Fallback for evolved modules without new parameters
            return self.evolved_module.create_test_attention_module(
                hidden_size=hidden_size,
                num_heads=num_heads, 
                num_kv_heads=num_kv_heads,
                head_dim=head_dim
            )


def load_and_patch_model(model_path: str, evolved_program_path: str, 
                        patch_attention: bool = True) -> Tuple[Any, Any]:
    """
    Load a model and optionally patch it with optimized attention
    
    Args:
        model_path: Path to MLX model
        evolved_program_path: Path to evolved attention program
        patch_attention: Whether to patch attention layers
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if not MLX_LM_AVAILABLE:
        raise ImportError("mlx_lm required for model loading")
    
    print(f"📥 Loading model: {model_path}")
    model, tokenizer = load(model_path)
    
    if patch_attention:
        print(f"🔧 Patching with optimized attention: {evolved_program_path}")
        wrapper = OptimizedAttentionWrapper(evolved_program_path)
        
        # Try to detect and patch attention layers
        # This is model-specific and may need adjustment for different architectures
        patched_count = _patch_model_attention(model, wrapper)
        print(f"✅ Patched {patched_count} attention layers")
    
    return model, tokenizer


def _patch_model_attention(model: nn.Module, wrapper: OptimizedAttentionWrapper) -> int:
    """
    Attempt to patch attention layers in a model
    This is a heuristic approach that works for common architectures
    
    Args:
        model: MLX model to patch
        wrapper: Optimized attention wrapper
        
    Returns:
        Number of layers patched
    """
    patched_count = 0
    
    # Common patterns for attention layer names
    attention_patterns = [
        'self_attn', 'attention', 'attn', 'multi_head_attention'
    ]
    
    def _recursive_patch(module, name_prefix=""):
        nonlocal patched_count
        
        for name, child in module.__dict__.items():
            if isinstance(child, nn.Module):
                full_name = f"{name_prefix}.{name}" if name_prefix else name
                
                # Check if this is an attention layer
                if any(pattern in name.lower() for pattern in attention_patterns):
                    try:
                        # Try to extract architecture details
                        if hasattr(child, 'hidden_size') and hasattr(child, 'num_heads'):
                            hidden_size = child.hidden_size
                            num_heads = child.num_heads
                            num_kv_heads = getattr(child, 'num_kv_heads', num_heads)
                            
                            # Create optimized replacement
                            optimized_attn = wrapper.create_optimized_attention(
                                hidden_size=hidden_size,
                                num_heads=num_heads,
                                num_kv_heads=num_kv_heads
                            )
                            
                            # Replace the attention layer
                            setattr(module, name, optimized_attn)
                            patched_count += 1
                            print(f"    Patched: {full_name}")
                            
                    except Exception as e:
                        print(f"    ⚠️  Failed to patch {full_name}: {str(e)}")
                
                # Recursively check children
                _recursive_patch(child, full_name)
    
    _recursive_patch(model)
    return patched_count


def compare_attention_performance(model_path: str, evolved_program_path: str,
                                prompt: str = "Write a Python function that",
                                max_tokens: int = 100, runs: int = 3) -> Dict[str, Any]:
    """
    Compare performance between standard and optimized attention
    
    Args:
        model_path: Path to MLX model
        evolved_program_path: Path to evolved attention program  
        prompt: Test prompt for generation
        max_tokens: Maximum tokens to generate
        runs: Number of benchmark runs
        
    Returns:
        Performance comparison results
    """
    
    if not MLX_LM_AVAILABLE:
        raise ImportError("mlx_lm required for performance comparison")
    
    print(f"⚖️  Comparing attention performance...")
    print(f"   Model: {model_path}")
    print(f"   Prompt: '{prompt}'")
    print(f"   Max tokens: {max_tokens}")
    
    results = {
        "model_path": model_path,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "runs": runs
    }
    
    # Test standard attention
    print(f"\n📊 Testing standard attention...")
    standard_model, tokenizer = load(model_path)
    standard_times = []
    
    for run in range(runs):
        start_time = time.time()
        try:
            response = generate(standard_model, tokenizer, prompt, 
                              max_tokens=max_tokens, verbose=False)
            end_time = time.time()
            
            run_time = end_time - start_time
            standard_times.append(run_time)
            
            tokens_generated = len(response.split()) - len(prompt.split())
            tokens_per_sec = tokens_generated / run_time if run_time > 0 else 0
            
            print(f"   Run {run+1}: {run_time:.2f}s ({tokens_per_sec:.1f} tokens/sec)")
            
        except Exception as e:
            print(f"   Run {run+1} failed: {str(e)}")
            standard_times.append(float('inf'))
    
    # Test optimized attention
    print(f"\n🚀 Testing optimized attention...")
    optimized_model, tokenizer = load_and_patch_model(model_path, evolved_program_path)
    optimized_times = []
    
    for run in range(runs):
        start_time = time.time()
        try:
            response = generate(optimized_model, tokenizer, prompt,
                              max_tokens=max_tokens, verbose=False)
            end_time = time.time()
            
            run_time = end_time - start_time
            optimized_times.append(run_time)
            
            tokens_generated = len(response.split()) - len(prompt.split())
            tokens_per_sec = tokens_generated / run_time if run_time > 0 else 0
            
            print(f"   Run {run+1}: {run_time:.2f}s ({tokens_per_sec:.1f} tokens/sec)")
            
        except Exception as e:
            print(f"   Run {run+1} failed: {str(e)}")
            optimized_times.append(float('inf'))
    
    # Calculate comparison
    valid_standard = [t for t in standard_times if t < float('inf')]
    valid_optimized = [t for t in optimized_times if t < float('inf')]
    
    if valid_standard and valid_optimized:
        avg_standard = sum(valid_standard) / len(valid_standard)
        avg_optimized = sum(valid_optimized) / len(valid_optimized)
        speedup = avg_standard / avg_optimized if avg_optimized > 0 else 0
        
        results.update({
            "standard_avg_time": avg_standard,
            "optimized_avg_time": avg_optimized,
            "speedup": speedup,
            "standard_successful_runs": len(valid_standard),
            "optimized_successful_runs": len(valid_optimized),
            "improvement": "Yes" if speedup > 1.05 else "Minimal" if speedup > 1.0 else "No"
        })
        
        print(f"\n📈 RESULTS:")
        print(f"   Standard attention: {avg_standard:.2f}s average")
        print(f"   Optimized attention: {avg_optimized:.2f}s average") 
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Improvement: {results['improvement']}")
        
    else:
        results["error"] = "Insufficient successful runs for comparison"
        print(f"\n❌ Comparison failed: insufficient successful runs")
    
    return results


def quick_demo(evolved_program_path: str, 
               model_path: str = "mlx-community/Qwen3-0.6B-bf16"):
    """
    Quick demonstration of optimized attention
    
    Args:
        evolved_program_path: Path to evolved attention program
        model_path: Model to test with
    """
    
    print("🚀 OpenEvolve Optimized Attention Demo")
    print("=" * 50)
    
    try:
        # Load model with optimized attention
        print(f"\n1️⃣  Loading model with optimized attention...")
        model, tokenizer = load_and_patch_model(model_path, evolved_program_path)
        
        # Test prompts
        test_prompts = [
            "Write a Python function that calculates fibonacci numbers:",
            "Explain machine learning in simple terms:",
            "Create a haiku about programming:"
        ]
        
        print(f"\n2️⃣  Testing text generation...")
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n   Test {i}: {prompt}")
            
            start_time = time.time()
            response = generate(model, tokenizer, prompt, max_tokens=50, verbose=False)
            end_time = time.time()
            
            generation_time = end_time - start_time
            tokens_generated = len(response.split()) - len(prompt.split())
            tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0
            
            print(f"   Response: {response[len(prompt):].strip()}")
            print(f"   Performance: {generation_time:.2f}s ({tokens_per_sec:.1f} tokens/sec)")
        
        print(f"\n✅ Demo complete! The optimized attention is working.")
        print(f"   Run the full benchmark for detailed performance comparisons.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        raise


def main():
    """Command-line interface for attention integration"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="MLX Attention Integration Helper")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Quick demonstration')
    demo_parser.add_argument('--evolved-program', required=True,
                           help='Path to evolved attention program')
    demo_parser.add_argument('--model', default='mlx-community/Qwen3-0.6B-bf16',
                           help='Model to test with')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare standard vs optimized')
    compare_parser.add_argument('--evolved-program', required=True,
                              help='Path to evolved attention program')
    compare_parser.add_argument('--model', default='mlx-community/Qwen3-0.6B-bf16',
                              help='Model to test with')
    compare_parser.add_argument('--prompt', default='Write a Python function that',
                              help='Test prompt')
    compare_parser.add_argument('--max-tokens', type=int, default=100,
                              help='Maximum tokens to generate')
    compare_parser.add_argument('--runs', type=int, default=3,
                              help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    if args.command == 'demo':
        quick_demo(args.evolved_program, args.model)
    elif args.command == 'compare':
        compare_attention_performance(
            args.model, args.evolved_program,
            args.prompt, args.max_tokens, args.runs
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
