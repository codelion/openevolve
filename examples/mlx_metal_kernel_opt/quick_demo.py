#!/usr/bin/env python3
"""
Quick Demo: AlphaEvolve Optimized Attention

Runs a quick demo showing performance differences.
"""

import os
import subprocess


def main():
    print("🎉 AlphaEvolve MLX Attention Demo")
    print("=" * 40)

    # Check dependencies
    try:
        import mlx
        import mlx_lm

        print("✅ Dependencies available")
    except ImportError as e:
        print(f"❌ Missing: {e}")
        print("   Run: pip install -r requirements.txt")
        return

    # Check for optimized program
    locations = ["openevolve_output/best/best_program.py", "best_program.py"]
    found = any(os.path.exists(loc) for loc in locations)

    if not found:
        print("❌ No optimized program found!")
        print("   Please run AlphaEvolve first.")
        return

    print(f"✅ Found optimized program")

    # Test cases
    tests = [
        ("Quick test", "The future of AI is", 500),
        ("Code generation", "def quicksort(arr):", 800),
        ("Reasoning", "To solve this step by step", 1600),
    ]

    print(f"\nRunning {len(tests)} comparison tests...\n")

    for i, (name, prompt, tokens) in enumerate(tests, 1):
        print(f"Test {i}/{len(tests)}: {name}")
        print(f"Prompt: '{prompt}'")
        print("-" * 30)

        cmd = [
            "python",
            "test_optimized_attention.py",
            "--prompt",
            prompt,
            "--max-tokens",
            str(tokens),
        ]

        try:
            subprocess.run(cmd, check=True)
            print("✅ Test completed")
        except subprocess.CalledProcessError:
            print("❌ Test failed")
        except KeyboardInterrupt:
            print("\n⚠️  Demo interrupted")
            break

        if i < len(tests):
            print("\n" + "=" * 40 + "\n")

    print("\n🎯 Demo completed!")
    print("💡 Run individual tests: python test_optimized_attention.py --prompt 'Your prompt'")


if __name__ == "__main__":
    main()
