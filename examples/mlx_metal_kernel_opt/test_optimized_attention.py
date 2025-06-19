#!/usr/bin/env python3
"""
Simple Test Script for Optimized MLX Attention

This script demonstrates how to monkey patch the official mlx-lm library
with the AlphaEvolve optimized attention kernel and shows the performance
difference on a test prompt.

Usage:
    python test_optimized_attention.py [path_to_best_program.py]

    If no path is provided, it will use the default best_program.py from
    openevolve_output/best/
"""

import os
import sys
import time
import argparse
import subprocess
import tempfile
from typing import Optional, Dict, Any
import traceback


def find_best_program() -> Optional[str]:
    """Find the best_program.py file in the expected location"""
    # Default location
    default_path = os.path.join(
        os.path.dirname(__file__), "openevolve_output", "best", "best_program.py"
    )

    if os.path.exists(default_path):
        return default_path

    # Alternative locations to check
    alternatives = [
        "best_program.py",
        "openevolve_output/best/best_program.py",
        "../best_program.py",
    ]

    for alt in alternatives:
        if os.path.exists(alt):
            return alt

    return None


def load_custom_attention_class(program_path: str):
    """Load the CustomGQAAttention class from the evolved program"""
    print(f"📁 Loading optimized attention from: {program_path}")

    try:
        # Read the program
        with open(program_path, "r") as f:
            program_text = f.read()

        # Setup execution environment
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from typing import Optional, Tuple, Any

        exec_globals = {
            "__builtins__": __builtins__,
            "mx": mx,
            "nn": nn,
            "np": np,
            "time": time,
            "Optional": Optional,
            "Tuple": Tuple,
            "Any": Any,
        }

        # Add mlx_lm imports for RoPE
        try:
            exec_globals["mlx_lm"] = __import__("mlx_lm")
        except ImportError:
            print("⚠️  Could not import mlx_lm, RoPE may not work")

        # Execute the program
        exec(program_text, exec_globals)

        # Extract the custom attention class
        custom_class = exec_globals.get("CustomGQAAttention")
        if custom_class is None:
            raise ValueError("CustomGQAAttention class not found in program")

        print("✅ Successfully loaded CustomGQAAttention class")
        return custom_class

    except Exception as e:
        print(f"❌ Failed to load custom attention: {e}")
        traceback.print_exc()
        return None


def apply_monkey_patch(custom_attention_class):
    """Apply monkey patch to replace Qwen3 attention with custom implementation"""
    print("🔧 Applying monkey patch to mlx-lm...")

    try:
        import mlx_lm.models.qwen3 as qwen3_module

        # Store original attention class
        original_attention = qwen3_module.Attention

        # Replace with custom implementation
        qwen3_module.Attention = custom_attention_class

        print("✅ Successfully applied monkey patch")
        return original_attention

    except ImportError as e:
        print(f"❌ Could not import mlx_lm.models.qwen3: {e}")
        print("   Make sure mlx-lm is installed: pip install mlx-lm")
        return None
    except Exception as e:
        print(f"❌ Failed to apply monkey patch: {e}")
        return None


def remove_monkey_patch(original_attention):
    """Remove the monkey patch and restore original attention"""
    if original_attention is None:
        return

    try:
        import mlx_lm.models.qwen3 as qwen3_module

        qwen3_module.Attention = original_attention
        print("✅ Removed monkey patch")
    except ImportError:
        pass


def run_mlx_lm_generation(
    prompt: str,
    max_tokens: int = 1000,
    model: str = "mlx-community/Qwen3-0.6B-bf16",
    debug: bool = False,
) -> Dict[str, Any]:
    """Run mlx-lm generation and parse the output"""
    print(f"🧪 Running generation with prompt: '{prompt[:50]}...'")

    try:
        # Also need to update the deprecated command format
        cmd = [
            "python",
            "-m",
            "mlx_lm",
            "generate",  # Updated format
            "--model",
            model,
            "--prompt",
            prompt,
            "--max-tokens",
            str(max_tokens),
            "--temp",
            "0.1",  # Low temperature for consistent results
        ]

        if debug:
            print(f"🔧 Running command: {' '.join(cmd)}")

        # Run generation
        start_time = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        end_time = time.perf_counter()

        if debug:
            print(f"📤 Command output:")
            print(f"Return code: {result.returncode}")
            print(f"STDOUT length: {len(result.stdout)}")
            print(f"STDERR length: {len(result.stderr)}")
            if result.stdout:
                print("First 500 chars of stdout:")
                print(result.stdout[:500])
            if result.stderr:
                print("STDERR:")
                print(result.stderr[:500])

        if result.returncode != 0:
            print(f"❌ Generation failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}")
            return {"success": False, "error": result.stderr}

        # Parse output
        output_lines = result.stdout.strip().split("\n")

        prompt_tokens = 0
        generation_tokens = 0
        prompt_speed = 0.0
        generation_speed = 0.0
        peak_memory = 0.0
        generated_text = ""

        # Find the generated text (everything after the prompt)
        capture_text = False
        found_prompt_stats = False
        found_generation_stats = False

        for line in output_lines:
            if debug:
                print(f"Parsing line: {line[:100]}")

            if line.startswith("=========="):
                capture_text = True
                continue
            elif (
                capture_text
                and line.strip()
                and not line.startswith("Prompt:")
                and not line.startswith("Generation:")
                and not line.startswith("Peak memory:")
            ):
                generated_text += line + "\n"
            elif "Prompt:" in line and "tokens-per-sec" in line:
                try:
                    # Parse: "Prompt: 9 tokens, 245.085 tokens-per-sec"
                    parts = line.split(",")
                    prompt_tokens = int(parts[0].split(":")[1].strip().split()[0])
                    prompt_speed = float(parts[1].strip().split()[0])
                    found_prompt_stats = True
                    if debug:
                        print(f"Found prompt stats: {prompt_tokens} tokens, {prompt_speed} tok/sec")
                except (ValueError, IndexError) as e:
                    if debug:
                        print(f"Failed to parse prompt line: {e}")
            elif "Generation:" in line and "tokens-per-sec" in line:
                try:
                    # Parse: "Generation: 82 tokens, 77.143 tokens-per-sec"
                    parts = line.split(",")
                    generation_tokens = int(parts[0].split(":")[1].strip().split()[0])
                    generation_speed = float(parts[1].strip().split()[0])
                    found_generation_stats = True
                    if debug:
                        print(
                            f"Found generation stats: {generation_tokens} tokens, {generation_speed} tok/sec"
                        )
                except (ValueError, IndexError) as e:
                    if debug:
                        print(f"Failed to parse generation line: {e}")
            elif "Peak memory:" in line:
                try:
                    memory_str = line.split(":")[1].strip()
                    if "GB" in memory_str:
                        peak_memory = float(memory_str.replace("GB", "").strip())
                    elif "MB" in memory_str:
                        peak_memory = float(memory_str.replace("MB", "").strip()) / 1024
                    if debug:
                        print(f"Found memory: {peak_memory} GB")
                except (ValueError, IndexError) as e:
                    if debug:
                        print(f"Failed to parse memory line: {e}")

        # Check if we got meaningful results
        if not found_generation_stats or generation_tokens == 0:
            print("⚠️  No generation statistics found in output")
            if debug:
                print(f"found_prompt_stats: {found_prompt_stats}")
                print(f"found_generation_stats: {found_generation_stats}")
                print(f"generation_tokens: {generation_tokens}")
                print("Full output for debugging:")
                print(result.stdout)
            return {"success": False, "error": "No generation statistics found"}

        result_dict = {
            "success": True,
            "prompt_tokens": prompt_tokens,
            "generation_tokens": generation_tokens,
            "prompt_speed": prompt_speed,
            "generation_speed": generation_speed,
            "peak_memory": peak_memory,
            "total_time": end_time - start_time,
            "generated_text": generated_text.strip(),
            "full_output": result.stdout,
        }

        if debug:
            print(f"Parsed result: {result_dict}")

        return result_dict

    except subprocess.TimeoutExpired:
        print("⏰ Generation timed out after 120 seconds")
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        if debug:
            traceback.print_exc()
        return {"success": False, "error": str(e)}


def run_comparison_test(
    prompt: str, custom_attention_class, max_tokens: int = 1000, debug: bool = False
):
    """Run comparison test between standard and optimized attention"""
    print(f"\n{'='*60}")
    print("🔬 ATTENTION COMPARISON TEST")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_tokens}")
    print()

    # Test 1: Standard attention
    print("📊 Testing STANDARD attention...")
    standard_result = run_mlx_lm_generation(prompt, max_tokens, debug=debug)

    if not standard_result.get("success", False):
        print("❌ Standard attention test failed")
        if debug and "error" in standard_result:
            print(f"   Error: {standard_result['error']}")
        print("\n🔧 Troubleshooting tips:")
        print("   • Check that mlx-lm is installed: pip install mlx-lm")
        print("   • Try a shorter prompt or fewer tokens")
        print("   • Run with --debug flag for more info")
        print("   • Check if the model downloads successfully")
        return

    print(f"✅ Standard Results:")
    print(f"   Decode Speed: {standard_result['generation_speed']:.1f} tokens/sec")
    print(f"   Memory Usage: {standard_result['peak_memory']:.2f} GB")
    print(f"   Total Time: {standard_result['total_time']:.2f} seconds")
    print(f"   Generated: {standard_result['generation_tokens']} tokens")

    # Check if we have valid results
    if standard_result["generation_tokens"] == 0:
        print("⚠️  Warning: Standard attention generated 0 tokens")
        print("   This might indicate an issue with the model or prompt")
        print("   Generated text preview:")
        print(f"   '{standard_result['generated_text'][:100]}'")

        # Ask user if they want to continue
        try:
            response = input("\n❓ Continue with optimized test anyway? (y/n): ").lower()
            if response != "y":
                print("Test cancelled")
                return
        except KeyboardInterrupt:
            print("\nTest cancelled")
            return

    # Apply monkey patch
    original_attention = apply_monkey_patch(custom_attention_class)
    if original_attention is None:
        print("❌ Failed to apply monkey patch")
        return

    try:
        # Test 2: Optimized attention
        print("\n📊 Testing OPTIMIZED attention...")
        optimized_result = run_mlx_lm_generation(prompt, max_tokens, debug=debug)

        if not optimized_result.get("success", False):
            print("❌ Optimized attention test failed")
            if debug and "error" in optimized_result:
                print(f"   Error: {optimized_result['error']}")
            return

        print(f"✅ Optimized Results:")
        print(f"   Decode Speed: {optimized_result['generation_speed']:.1f} tokens/sec")
        print(f"   Memory Usage: {optimized_result['peak_memory']:.2f} GB")
        print(f"   Total Time: {optimized_result['total_time']:.2f} seconds")
        print(f"   Generated: {optimized_result['generation_tokens']} tokens")

        # Calculate improvements (handle division by zero)
        if standard_result["generation_speed"] > 0:
            speed_improvement = (
                (optimized_result["generation_speed"] - standard_result["generation_speed"])
                / standard_result["generation_speed"]
            ) * 100
        else:
            speed_improvement = 0.0
            print("⚠️  Cannot calculate speed improvement (standard speed was 0)")

        memory_change = optimized_result["peak_memory"] - standard_result["peak_memory"]

        if standard_result["total_time"] > 0:
            time_improvement = (
                (standard_result["total_time"] - optimized_result["total_time"])
                / standard_result["total_time"]
            ) * 100
        else:
            time_improvement = 0.0

        print(f"\n🚀 PERFORMANCE COMPARISON:")
        if standard_result["generation_speed"] > 0:
            print(f"   Speed Improvement: {speed_improvement:+.1f}%")
        else:
            print(
                f"   Speed Comparison: {standard_result['generation_speed']:.1f} → {optimized_result['generation_speed']:.1f} tokens/sec"
            )
        print(f"   Memory Change: {memory_change:+.2f} GB")
        print(f"   Time Improvement: {time_improvement:+.1f}%")

        if speed_improvement > 5:
            print("🎯 SIGNIFICANT IMPROVEMENT achieved!")
        elif speed_improvement > 0:
            print("📈 Modest improvement achieved")
        elif standard_result["generation_speed"] == 0 and optimized_result["generation_speed"] > 0:
            print("🔥 Optimized version works where standard failed!")
        else:
            print("⚠️  No improvement or regression")

        # Show generated text comparison
        print(f"\n📝 GENERATED TEXT COMPARISON:")
        std_text = (
            standard_result["generated_text"][:200]
            if standard_result["generated_text"]
            else "[No text generated]"
        )
        opt_text = (
            optimized_result["generated_text"][:200]
            if optimized_result["generated_text"]
            else "[No text generated]"
        )

        print(f"Standard: {std_text}...")
        print(f"Optimized: {opt_text}...")

        if standard_result["generated_text"] and optimized_result["generated_text"]:
            if standard_result["generated_text"][:100] == optimized_result["generated_text"][:100]:
                print("✅ Generated text is identical (good!)")
            else:
                print("⚠️  Generated text differs (check randomness/temperature)")
        elif not standard_result["generated_text"] and not optimized_result["generated_text"]:
            print("⚠️  Both versions generated no text")
        else:
            print("ℹ️  Different text generation behavior")

    finally:
        # Always remove monkey patch
        remove_monkey_patch(original_attention)


def main():
    parser = argparse.ArgumentParser(description="Test optimized MLX attention kernel")
    parser.add_argument("program_path", nargs="?", help="Path to best_program.py")
    parser.add_argument(
        "--prompt", default="The future of artificial intelligence is", help="Test prompt"
    )
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-bf16", help="Model to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Find program path
    if args.program_path:
        program_path = args.program_path
    else:
        program_path = find_best_program()

    if not program_path or not os.path.exists(program_path):
        print("❌ Could not find best_program.py")
        print("   Please provide the path to the optimized program:")
        print("   python test_optimized_attention.py path/to/best_program.py")
        print("\n   Or make sure you have run AlphaEvolve and have results in:")
        print("   openevolve_output/best/best_program.py")
        sys.exit(1)

    print("🚀 MLX Optimized Attention Tester")
    print(f"Using program: {program_path}")
    print(f"Model: {args.model}")
    if args.debug:
        print("🐛 Debug mode enabled")

    # Load custom attention
    custom_attention_class = load_custom_attention_class(program_path)
    if custom_attention_class is None:
        sys.exit(1)

    # Check if mlx-lm is available
    try:
        import mlx_lm

        print("✅ mlx-lm is available")
    except ImportError:
        print("❌ mlx-lm is not installed")
        print("   Please install it: pip install mlx-lm")
        sys.exit(1)

    # Run comparison test
    run_comparison_test(args.prompt, custom_attention_class, args.max_tokens, debug=args.debug)

    print(f"\n{'='*60}")
    print("✅ Test completed!")
    print("💡 To test with a different prompt:")
    print(f"   python {sys.argv[0]} --prompt 'Your custom prompt here'")
    print("💡 For debugging: add --debug flag")
    print("💡 For help: python test_optimized_attention.py --help")


if __name__ == "__main__":
    main()
