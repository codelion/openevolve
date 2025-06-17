#!/usr/bin/env python3
"""
MLX Metal Kernel Optimization Demo

This script demonstrates how to integrate Metal kernel optimizations with mlx-lm
for improved transformer performance on Apple Silicon. It shows before/after
comparisons and provides easy integration examples.

Usage:
    python demo_integration.py --model qwen2.5-0.5b --enable-optimization
    python demo_integration.py --model llama-3.2-1b --benchmark-only
    python demo_integration.py --quick-test
"""

import argparse
import time
import sys
import os
from pathlib import Path
from typing import Optional, List
import warnings

# Add integration to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
except ImportError:
    print("❌ MLX and MLX-LM are required. Install with:")
    print("   pip install mlx mlx-lm")
    sys.exit(1)

# Import our optimizations
from integration import (
    patch_mlx_lm, 
    unpatch_mlx_lm, 
    get_integration_status,
    benchmark_optimization,
    quick_benchmark
)


class MLXOptimizationDemo:
    """
    Comprehensive demonstration of MLX Metal kernel optimizations.
    """
    
    def __init__(self, enable_debug: bool = True):
        self.enable_debug = enable_debug
        self.model = None
        self.tokenizer = None
        
        # Popular models for testing
        self.test_models = {
            'qwen2.5-0.5b': 'mlx-community/Qwen2.5-0.5B-Instruct-4bit',
            'qwen2.5-1.5b': 'mlx-community/Qwen2.5-1.5B-Instruct-4bit',
            'llama-3.2-1b': 'mlx-community/Llama-3.2-1B-Instruct-4bit',
            'llama-3.2-3b': 'mlx-community/Llama-3.2-3B-Instruct-4bit',
            'gemma-2b': 'mlx-community/gemma-2b-it-4bit',
            'phi-3-mini': 'mlx-community/Phi-3-mini-4k-instruct-4bit',
        }
        
        self.test_prompts = [
            "Explain the concept of attention mechanisms in transformers.",
            "Write a Python function to calculate the Fibonacci sequence.",
            "What are the benefits of using Apple Silicon for machine learning?",
            "Describe the differences between GQA and standard multi-head attention.",
        ]

    def print_header(self, title: str):
        """Print a formatted header"""
        print("\n" + "=" * 70)
        print(f"🚀 {title}")
        print("=" * 70)

    def print_section(self, title: str):
        """Print a formatted section header"""
        print(f"\n📋 {title}")
        print("-" * 50)

    def load_model(self, model_key: str) -> bool:
        """Load a model for testing"""
        if model_key not in self.test_models:
            print(f"❌ Unknown model key: {model_key}")
            print(f"Available models: {list(self.test_models.keys())}")
            return False
        
        model_path = self.test_models[model_key]
        
        try:
            print(f"📥 Loading model: {model_path}")
            self.model, self.tokenizer = load(model_path)
            print(f"✅ Model loaded successfully")
            
            # Print model info
            if hasattr(self.model, 'args'):
                args = self.model.args
                print(f"   📊 Architecture: {getattr(args, 'num_attention_heads', 'Unknown')} heads, "
                      f"{getattr(args, 'num_key_value_heads', 'Unknown')} KV heads")
                print(f"   📏 Hidden size: {getattr(args, 'hidden_size', 'Unknown')}")
                print(f"   🧠 Head dim: {getattr(args, 'head_dim', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

    def generate_text(self, prompt: str, max_tokens: int = 50, temp: float = 0.7) -> tuple[str, float]:
        """Generate text and measure time"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded")
        
        start_time = time.perf_counter()
        
        try:
            response = generate(
                self.model, 
                self.tokenizer, 
                prompt=prompt, 
                max_tokens=max_tokens,
                temp=temp,
                verbose=False
            )
            
            # Force evaluation
            mx.eval(response)
            mx.synchronize()
            
            end_time = time.perf_counter()
            generation_time = end_time - start_time
            
            return response, generation_time
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return "", 0.0

    def benchmark_generation(self, model_key: str, num_runs: int = 3):
        """Benchmark text generation with and without optimizations"""
        
        self.print_header(f"Generation Benchmark: {model_key}")
        
        if not self.load_model(model_key):
            return
        
        prompt = self.test_prompts[0]  # Use first prompt for consistency
        max_tokens = 100
        
        # Test without optimizations
        self.print_section("Standard MLX-LM Performance")
        standard_times = []
        
        print(f"🔄 Running {num_runs} generations without optimizations...")
        for i in range(num_runs):
            response, gen_time = self.generate_text(prompt, max_tokens)
            standard_times.append(gen_time)
            print(f"   Run {i+1}: {gen_time:.2f}s ({len(response.split())} tokens)")
        
        avg_standard_time = sum(standard_times) / len(standard_times)
        print(f"⏱️  Average time: {avg_standard_time:.2f}s")
        
        # Test with optimizations
        self.print_section("Optimized Metal Kernel Performance")
        
        print("🔧 Applying Metal kernel optimizations...")
        patched_count = patch_mlx_lm(enable_debug=self.enable_debug)
        print(f"✅ Patched {patched_count} models")
        
        optimized_times = []
        
        print(f"⚡ Running {num_runs} generations with optimizations...")
        try:
            for i in range(num_runs):
                response, gen_time = self.generate_text(prompt, max_tokens)
                optimized_times.append(gen_time)
                print(f"   Run {i+1}: {gen_time:.2f}s ({len(response.split())} tokens)")
            
            avg_optimized_time = sum(optimized_times) / len(optimized_times)
            print(f"⏱️  Average time: {avg_optimized_time:.2f}s")
            
            # Calculate improvement
            speedup = avg_standard_time / avg_optimized_time
            improvement = ((avg_standard_time - avg_optimized_time) / avg_standard_time) * 100
            
            self.print_section("Performance Comparison")
            print(f"🚀 Speedup: {speedup:.2f}x")
            print(f"📈 Improvement: {improvement:.1f}%")
            print(f"⏰ Time saved: {avg_standard_time - avg_optimized_time:.2f}s per generation")
            
            # Show optimization stats
            status = get_integration_status()
            opt_stats = status.get('optimizer_stats', {})
            optimization_rate = opt_stats.get('optimization_rate', 0.0)
            print(f"📊 Optimization rate: {optimization_rate:.1%}")
            
        finally:
            # Clean up
            unpatch_mlx_lm(enable_debug=self.enable_debug)
            print("🧹 Removed optimizations")

    def interactive_demo(self, model_key: str):
        """Interactive demonstration with user prompts"""
        
        self.print_header(f"Interactive Demo: {model_key}")
        
        if not self.load_model(model_key):
            return
        
        print("🎮 Interactive mode - Enter prompts to test optimization")
        print("   Type 'optimize' to enable optimizations")
        print("   Type 'standard' to disable optimizations") 
        print("   Type 'status' to check optimization status")
        print("   Type 'quit' to exit")
        
        optimized = False
        
        while True:
            try:
                user_input = input("\n💬 Your prompt (or command): ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'optimize':
                    if not optimized:
                        patch_mlx_lm(enable_debug=self.enable_debug)
                        optimized = True
                        print("✅ Metal kernel optimizations enabled")
                    else:
                        print("⚠️ Optimizations already enabled")
                    continue
                elif user_input.lower() == 'standard':
                    if optimized:
                        unpatch_mlx_lm(enable_debug=self.enable_debug)
                        optimized = False
                        print("✅ Using standard MLX implementation")
                    else:
                        print("⚠️ Already using standard implementation")
                    continue
                elif user_input.lower() == 'status':
                    status = get_integration_status()
                    print(f"🔧 Optimizations enabled: {status['is_patched']}")
                    if status['optimizer_stats']:
                        stats = status['optimizer_stats']
                        print(f"📊 Total calls: {stats.get('total_calls', 0)}")
                        print(f"⚡ Optimized calls: {stats.get('optimized_calls', 0)}")
                        print(f"📈 Optimization rate: {stats.get('optimization_rate', 0):.1%}")
                    continue
                elif not user_input:
                    continue
                
                # Generate response
                mode = "⚡ Optimized" if optimized else "🔄 Standard"
                print(f"\n{mode} Generation:")
                
                response, gen_time = self.generate_text(user_input, max_tokens=150)
                
                print(f"🤖 Response ({gen_time:.2f}s):")
                print(f"{response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Clean up
        if optimized:
            unpatch_mlx_lm(enable_debug=self.enable_debug)

    def quick_comparison(self):
        """Quick side-by-side comparison"""
        
        self.print_header("Quick Optimization Comparison")
        
        # Use a smaller model for quick testing
        model_key = 'qwen2.5-0.5b'
        if not self.load_model(model_key):
            return
        
        prompt = "Write a short poem about machine learning."
        max_tokens = 80
        
        print(f"📝 Prompt: {prompt}")
        print(f"🎯 Max tokens: {max_tokens}")
        
        # Standard generation
        print("\n🔄 Standard MLX-LM:")
        standard_response, standard_time = self.generate_text(prompt, max_tokens)
        standard_memory = mx.get_active_memory() / 1e9
        
        print(f"⏱️  Time: {standard_time:.2f}s")
        print(f"💾 Memory: {standard_memory:.2f}GB")
        print(f"📝 Response:\n{standard_response}")
        
        # Optimized generation
        print("\n⚡ With Metal Kernel Optimization:")
        patch_mlx_lm(enable_debug=False)
        
        try:
            optimized_response, optimized_time = self.generate_text(prompt, max_tokens)
            optimized_memory = mx.get_active_memory() / 1e9
            
            print(f"⏱️  Time: {optimized_time:.2f}s")
            print(f"💾 Memory: {optimized_memory:.2f}GB")
            print(f"📝 Response:\n{optimized_response}")
            
            # Show comparison
            speedup = standard_time / optimized_time if optimized_time > 0 else 1.0
            memory_diff = standard_memory - optimized_memory
            
            print("\n📊 Comparison:")
            print(f"🚀 Speedup: {speedup:.2f}x")
            print(f"💾 Memory difference: {memory_diff:.2f}GB")
            
            status = get_integration_status()
            opt_stats = status.get('optimizer_stats', {})
            print(f"📈 Optimization rate: {opt_stats.get('optimization_rate', 0):.1%}")
            
        finally:
            unpatch_mlx_lm(enable_debug=False)

    def run_comprehensive_test(self):
        """Run comprehensive test across multiple models"""
        
        self.print_header("Comprehensive Metal Kernel Test Suite")
        
        # Test available models
        available_models = []
        for model_key in ['qwen2.5-0.5b', 'llama-3.2-1b']:
            print(f"\n🔍 Testing model availability: {model_key}")
            if self.load_model(model_key):
                available_models.append(model_key)
                print(f"✅ {model_key} is available")
            else:
                print(f"❌ {model_key} is not available")
        
        if not available_models:
            print("❌ No models available for testing")
            return
        
        # Run tests
        for model_key in available_models:
            self.benchmark_generation(model_key, num_runs=2)
        
        # Run attention-level benchmarking
        print("\n🧪 Running attention kernel benchmarks...")
        try:
            benchmark_results = benchmark_optimization(
                model_name="qwen3",
                seq_lengths=[256, 512, 1024],
                warmup_runs=2,
                benchmark_runs=3,
                save_results=True
            )
            
            print("✅ Kernel benchmarks completed")
            
        except Exception as e:
            print(f"⚠️ Kernel benchmark failed: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="MLX Metal Kernel Optimization Demo")
    parser.add_argument('--model', choices=['qwen2.5-0.5b', 'qwen2.5-1.5b', 'llama-3.2-1b', 'llama-3.2-3b'], 
                       default='qwen2.5-0.5b', help='Model to test')
    parser.add_argument('--quick-test', action='store_true', help='Run quick comparison test')
    parser.add_argument('--benchmark-only', action='store_true', help='Run benchmark only')
    parser.add_argument('--interactive', action='store_true', help='Run interactive demo')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive test suite')
    parser.add_argument('--kernel-benchmark', action='store_true', help='Run kernel-level benchmark only')
    parser.add_argument('--disable-debug', action='store_true', help='Disable debug output')
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = MLXOptimizationDemo(enable_debug=not args.disable_debug)
    
    try:
        if args.quick_test:
            demo.quick_comparison()
        elif args.benchmark_only:
            demo.benchmark_generation(args.model)
        elif args.interactive:
            demo.interactive_demo(args.model)
        elif args.comprehensive:
            demo.run_comprehensive_test()
        elif args.kernel_benchmark:
            quick_benchmark(enable_debug=not args.disable_debug)
        else:
            # Default: show quick test and offer options
            demo.quick_comparison()
            
            print("\n🎯 What would you like to do next?")
            print("1. Interactive demo")
            print("2. Full benchmark")
            print("3. Kernel-level benchmark")
            print("4. Exit")
            
            choice = input("\nChoose an option (1-4): ").strip()
            
            if choice == '1':
                demo.interactive_demo(args.model)
            elif choice == '2':
                demo.benchmark_generation(args.model)
            elif choice == '3':
                quick_benchmark(enable_debug=not args.disable_debug)
            else:
                print("👋 Goodbye!")
                
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        if not args.disable_debug:
            import traceback
            traceback.print_exc()
    finally:
        # Ensure cleanup
        try:
            unpatch_mlx_lm(enable_debug=False)
        except:
            pass


if __name__ == "__main__":
    main()
