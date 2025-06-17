#!/usr/bin/env python3
"""
Simple Usage Examples for MLX Metal Kernel Optimization

This script shows the most common usage patterns for integrating Metal kernel
optimizations with existing mlx-lm workflows.
"""

import sys
from pathlib import Path

# Add integration to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import mlx.core as mx
    from mlx_lm import load, generate
except ImportError:
    print("❌ Please install MLX and MLX-LM:")
    print("   pip install mlx mlx-lm")
    sys.exit(1)

from integration import patch_mlx_lm, unpatch_mlx_lm, get_integration_status


def example_1_basic_usage():
    """Example 1: Basic usage with automatic optimization"""
    print("🚀 Example 1: Basic Usage with Automatic Optimization")
    print("=" * 60)
    
    # Apply optimizations before loading model
    print("1. Applying Metal kernel optimizations...")
    patched_count = patch_mlx_lm(enable_debug=True)
    print(f"   ✅ Patched {patched_count} models")
    
    try:
        # Load model (optimizations will be applied automatically)
        print("\n2. Loading model...")
        model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        print("   ✅ Model loaded with optimizations")
        
        # Generate text (uses optimized kernels automatically)
        print("\n3. Generating text with optimizations...")
        prompt = "Explain how attention mechanisms work in transformers."
        response = generate(model, tokenizer, prompt=prompt, max_tokens=100, temp=0.7)
        
        print(f"   📝 Prompt: {prompt}")
        print(f"   🤖 Response: {response}")
        
        # Check optimization stats
        status = get_integration_status()
        opt_stats = status.get('optimizer_stats', {})
        print(f"\n📊 Optimization Stats:")
        print(f"   Total calls: {opt_stats.get('total_calls', 0)}")
        print(f"   Optimized calls: {opt_stats.get('optimized_calls', 0)}")
        print(f"   Optimization rate: {opt_stats.get('optimization_rate', 0):.1%}")
        
    finally:
        # Remove optimizations when done
        print("\n4. Cleaning up...")
        unpatch_mlx_lm(enable_debug=True)
        print("   ✅ Optimizations removed")


def example_2_context_manager():
    """Example 2: Using context manager pattern"""
    print("\n🚀 Example 2: Context Manager Pattern")
    print("=" * 60)
    
    class OptimizedMLX:
        """Context manager for temporary optimizations"""
        
        def __init__(self, enable_debug=False):
            self.enable_debug = enable_debug
            self.patched_count = 0
            
        def __enter__(self):
            print("🔧 Applying optimizations...")
            self.patched_count = patch_mlx_lm(enable_debug=self.enable_debug)
            print(f"   ✅ Patched {self.patched_count} models")
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            print("🧹 Removing optimizations...")
            unpatch_mlx_lm(enable_debug=self.enable_debug)
            print("   ✅ Optimizations removed")
    
    # Use optimizations only within this block
    with OptimizedMLX(enable_debug=True):
        model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        
        prompt = "What are the benefits of using Apple Silicon for AI?"
        response = generate(model, tokenizer, prompt=prompt, max_tokens=80)
        
        print(f"📝 Generated with optimizations: {response}")
    
    print("✅ Optimizations automatically removed")


def example_3_before_after_comparison():
    """Example 3: Before/after performance comparison"""
    print("\n🚀 Example 3: Before/After Performance Comparison")
    print("=" * 60)
    
    import time
    
    # Load model first
    print("Loading model...")
    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    
    prompt = "Write a Python function to sort a list."
    max_tokens = 100
    
    # Test without optimizations
    print("\n1. Testing WITHOUT optimizations...")
    start_time = time.perf_counter()
    response_standard = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    standard_time = time.perf_counter() - start_time
    
    print(f"   ⏱️  Time: {standard_time:.2f}s")
    print(f"   📝 Response length: {len(response_standard.split())} words")
    
    # Test with optimizations
    print("\n2. Testing WITH optimizations...")
    patch_mlx_lm(enable_debug=False)
    
    try:
        start_time = time.perf_counter()
        response_optimized = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
        optimized_time = time.perf_counter() - start_time
        
        print(f"   ⏱️  Time: {optimized_time:.2f}s")
        print(f"   📝 Response length: {len(response_optimized.split())} words")
        
        # Show improvement
        speedup = standard_time / optimized_time if optimized_time > 0 else 1.0
        improvement = ((standard_time - optimized_time) / standard_time) * 100
        
        print(f"\n📊 Performance Improvement:")
        print(f"   🚀 Speedup: {speedup:.2f}x")
        print(f"   📈 Improvement: {improvement:.1f}%")
        
    finally:
        unpatch_mlx_lm(enable_debug=False)


def example_4_custom_configuration():
    """Example 4: Custom optimization configuration"""
    print("\n🚀 Example 4: Custom Optimization Configuration")
    print("=" * 60)
    
    from integration import configure_optimizer
    
    # Configure optimizer with custom thresholds
    print("🔧 Configuring optimizer with custom settings...")
    configure_optimizer(
        enable_debug=True,
        min_seq_len=128,        # Lower threshold for short sequences
        max_seq_len=2048,       # Higher limit for long sequences
        gqa_ratio_min=3,        # Require at least 3:1 GQA ratio
        min_heads=16            # Require at least 16 heads
    )
    
    # Apply with custom configuration
    patched_count = patch_mlx_lm(enable_debug=True)
    print(f"✅ Applied custom optimizations to {patched_count} models")
    
    try:
        model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        
        # Test with different sequence lengths
        test_prompts = [
            "Short test.",  # Very short
            "This is a medium length prompt that should trigger optimization based on our custom settings.",  # Medium
            "This is a very long prompt " * 20 + " that tests our custom sequence length limits."  # Long
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{i}. Testing prompt length: {len(prompt.split())} words")
            response = generate(model, tokenizer, prompt=prompt, max_tokens=50)
            print(f"   ✅ Generated successfully")
        
        # Show final stats
        status = get_integration_status()
        opt_stats = status.get('optimizer_stats', {})
        print(f"\n📊 Final optimization rate: {opt_stats.get('optimization_rate', 0):.1%}")
        
    finally:
        unpatch_mlx_lm(enable_debug=True)


def example_5_selective_model_patching():
    """Example 5: Patching specific models only"""
    print("\n🚀 Example 5: Selective Model Patching")
    print("=" * 60)
    
    from integration.mlx_lm_integration import MLXLMIntegration
    
    # Create custom integration instance
    integration = MLXLMIntegration()
    
    # Patch only specific models
    print("🎯 Patching only Qwen models...")
    qwen_models = ['qwen3', 'qwen2']
    
    for model_name in qwen_models:
        success = integration.patch_model_attention(model_name, enable_debug=True)
        if success:
            print(f"   ✅ Patched {model_name}")
        else:
            print(f"   ❌ Failed to patch {model_name}")
    
    # Check what was patched
    status = integration.get_patch_status()
    print(f"\n📊 Patched modules: {status['patched_modules']}")
    
    try:
        # Test with Qwen model (should use optimizations)
        model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        response = generate(model, tokenizer, prompt="Test prompt", max_tokens=30)
        print(f"✅ Qwen model test: {response}")
        
    finally:
        # Clean up
        integration.unpatch_all(enable_debug=True)


def main():
    """Run all examples"""
    print("🧪 MLX Metal Kernel Optimization - Usage Examples")
    print("=" * 70)
    
    examples = [
        example_1_basic_usage,
        example_2_context_manager, 
        example_3_before_after_comparison,
        example_4_custom_configuration,
        example_5_selective_model_patching
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Example {i} failed: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(examples):
            input(f"\n⏸️  Press Enter to continue to Example {i+1}...")
    
    print("\n🎉 All examples completed!")
    print("\n💡 Integration Tips:")
    print("   1. Apply optimizations before loading models for best results")
    print("   2. Use context managers for temporary optimizations")
    print("   3. Check optimization stats to verify performance gains")
    print("   4. Configure thresholds based on your use case")
    print("   5. Always clean up optimizations when done")


if __name__ == "__main__":
    main()
