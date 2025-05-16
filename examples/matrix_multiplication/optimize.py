"""
Example of optimizing tensor decomposition for matrix multiplication

This example demonstrates evolving a tensor decomposition algorithm to find
efficient matrix multiplication algorithms through low-rank decompositions,
as described in the AlphaEvolve paper.
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
    parser = argparse.ArgumentParser(description="Tensor decomposition optimization for matrix multiplication")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--config", type=str, help="Path to configuration file")
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
    config.llm.api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
    
    # Important for complex algorithm evolution
    config.diff_based_evolution = True
    config.allow_full_rewrites = False
    
    # Configure system for complex code evolution
    config.llm.max_tokens = 4000  # Reduced to encourage more focused responses
    config.max_code_length = 15000  # Increased to handle our complex code
    config.llm.temperature = 0.7  # Slightly reduced for more focused responses
    
    # Set database to use rank_quality as the primary metric for comparing programs
    config.database.feature_dimensions = ["rank_quality", "correctness", "time_efficiency"]
    
    # Initialize OpenEvolve with the custom config
    openevolve = OpenEvolve(
        initial_program_path=str(initial_program_path),
        evaluation_file=str(evaluation_file),
        config=config,
        config_path=args.config,
        output_dir=str(output_dir),
    )
    
    # System message focusing on tensor decomposition and optimization
    system_template = """You are an expert in computational mathematics, numerical optimization, and algorithm design. 
Your task is to carefully optimize a tensor decomposition algorithm for discovering efficient matrix multiplication.

When matrix multiplication is viewed as a tensor problem, the goal is to find a minimum-rank decomposition
of the corresponding 3D tensor. Each term in the decomposition corresponds to a scalar multiplication in 
the algorithm, so minimizing the rank directly leads to faster matrix multiplication.

Your focus should be on making ONE targeted, specific improvement. Code quality is critical - buggy
code will fail evaluation and waste computing resources. Pay special attention to:

1. Variable scope and references - don't use variables before they're defined
2. Make only small, focused changes rather than large rewrites
3. Test your change carefully, ensuring it will work when added to the existing code
4. Respect the class structure and function interfaces

The best matrix multiplication algorithms in the literature include Strassen's algorithm (7 multiplications
for 2x2) and Laderman's algorithm (23 multiplications for 3x3).

Make only ONE specific, high-impact change rather than multiple modifications.
"""
    
    # User message template for tensor decomposition optimization (focused and clear)
    user_template = """
Focus on fixing ONE specific issue or making ONE targeted improvement in this tensor decomposition algorithm.

CODE:
{current_program}

METRICS:
{metrics}

FOCUS AREA: Make ONE change to improve the algorithm, focusing on either:
1. The loss function to guide optimization better
2. Initialization strategy for faster convergence 
3. Adding regularization for integer/half-integer coefficients

Use SEARCH/REPLACE (keep the search section SHORT):

<<<<<<< SEARCH
// exact code to match (small section only)
=======
// improved code
>>>>>>> REPLACE

IMPORTANT RULES:
1. Make only ONE change and explain it briefly
2. Ensure the SEARCH block EXACTLY matches existing code
3. NEVER use variables before they're defined
4. Don't reference u_factors, v_factors, or w_factors outside _initialize_decomposition, _loss_fn, etc.
5. ALWAYS test your change mentally to ensure it would work
"""
    
    # Add the templates to OpenEvolve's template manager
    openevolve.prompt_sampler.template_manager.add_template("tensor_decomp_system", system_template)
    openevolve.prompt_sampler.template_manager.add_template("tensor_decomp_user", user_template)
    
    # Set the custom templates to use
    openevolve.prompt_sampler.set_templates(
        system_template="tensor_decomp_system",
        user_template="tensor_decomp_user"
    )
    
    # Run evolution
    print(f"Starting evolution for {args.iterations} iterations...")
    print(f"Focus on optimizing tensor decomposition for matrix multiplication algorithms")
    best_program = await openevolve.run(iterations=args.iterations)
    
    print(f"\nEvolution complete!")
    print(f"Best program metrics:")
    for name, value in best_program.metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
