"""
Example of minimizing the ratio of maximum to minimum distance in a point set

This example demonstrates evolving an algorithm to find a set of points in 2D
that minimizes the ratio between the maximum and minimum pairwise distances.
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
    parser = argparse.ArgumentParser(
        description="Minimizing max/min distance ratio example"
    )
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--num-points", type=int, default=16, help="Number of points")
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
    
    # Update evaluation file with number of points
    with open(evaluation_file, "r") as f:
        eval_content = f.read()
    
    eval_content = eval_content.replace(
        "NUM_POINTS = 16", 
        f"NUM_POINTS = {args.num_points}"
    )
    
    with open(evaluation_file, "w") as f:
        f.write(eval_content)
    
    # Initialize OpenEvolve
    openevolve = OpenEvolve(
        initial_program_path=str(initial_program_path),
        evaluation_file=str(evaluation_file),
        output_dir=str(output_dir),
    )
    
    # Run evolution
    print(f"Starting evolution for {args.iterations} iterations...")
    print(f"Optimizing for {args.num_points} points in 2D...")
    best_program = await openevolve.run(iterations=args.iterations)
    
    print(f"\nEvolution complete!")
    print(f"Best program metrics:")
    for name, value in best_program.metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Generate visualization for the best solution
    print(f"\nGenerating visualization...")
    best_program_path = os.path.join(output_dir, "best", "best_program.python")
    
    if os.path.exists(best_program_path):
        # Import visualization module
        sys.path.append(str(current_dir))
        from visualize import visualize_solution
        
        # Visualize solution
        output_file = os.path.join(output_dir, "best_solution.png")
        visualize_solution(best_program_path, output_file)
        print(f"Visualization saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
