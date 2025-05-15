"""
Example of optimizing a matrix multiplication algorithm

This example demonstrates evolving a matrix multiplication algorithm to find
a more efficient implementation for specific sizes of matrices.
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openevolve import OpenEvolve
from openevolve.config import Config


async def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Matrix multiplication optimization example")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    args = parser.parse_args()
    
    # Set up paths
    current_dir = Path(__file__).parent
    initial_program_path = current_dir / "initial_program.py"
    evaluation_file = current_dir / "evaluate.py"
    output_dir = current_dir / args.output
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "evolution.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create configuration
    config = Config()
    config.max_iterations = args.iterations
    config.llm.primary_model = "gemini-2.0-flash-lite"
    config.llm.secondary_model = "gemini-2.0-flash"
    config.diff_based_evolution = True
    config.allow_full_rewrites = False
    
    # Initialize OpenEvolve
    openevolve = OpenEvolve(
        initial_program_path=str(initial_program_path),
        evaluation_file=str(evaluation_file),
        output_dir=str(output_dir),
    )
    
    # Run evolution
    print(f"Starting evolution for {args.iterations} iterations...")
    best_program = await openevolve.run(iterations=args.iterations)
    
    print(f"\nEvolution complete!")
    print(f"Best program metrics:")
    for name, value in best_program.metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
