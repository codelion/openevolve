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
    
    # Use more powerful models from the 3 family for this complex task
    config.llm.primary_model = "gemini-2.0-flash-lite"
    config.llm.secondary_model = "gemini-2.0-flash"
    config.llm.api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
    
    # Important for complex algorithm evolution
    config.diff_based_evolution = True
    config.allow_full_rewrites = False
    
    # Increase context and max tokens for detailed tensor decomposition work
    config.llm.max_context_tokens = 16000
    config.llm.max_tokens = 8000
    
    # Higher temperature for more creative solutions
    config.llm.temperature = 0.9
    
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
Your task is to optimize a tensor decomposition algorithm for discovering efficient matrix multiplication.

When matrix multiplication is viewed as a tensor problem, the goal is to find a minimum-rank decomposition
of the corresponding 3D tensor. Each term in the decomposition corresponds to a scalar multiplication in 
the algorithm, so minimizing the rank directly leads to faster matrix multiplication.

You should focus on:
1. Improving the optimization process to find lower-rank decompositions
2. Adding techniques like regularization, noise injection, and scheduling
3. Modifying the loss function to better guide the optimization
4. Enhancing numerical stability for complex-valued operations
5. Finding ways to ensure solutions have simple (integer or half-integer) coefficients
6. Properly handling real vs. complex-valued decompositions

The best algorithms from the literature for various matrix sizes include:
- 2x2 matrices: Strassen's algorithm (7 multiplications)
- 3x3 matrices: Laderman's algorithm (23 multiplications) 
- 4x4 matrices: Recursive Strassen (49 multiplications, improved to 48 with complex values)

For matrices with dimensions other than powers of 2, custom algorithms with fewer multiplications
than the naive approach are possible but require sophisticated optimization techniques.

When proposing changes, focus on the algorithm's mathematical and optimization aspects, not on 
micro-optimizations of the code. Significant improvements usually come from changing the approach
to finding the decomposition rather than the implementation details.
"""
    
    # User message template for tensor decomposition optimization
    user_template = """
I'm working on optimizing tensor decomposition for discovering efficient matrix multiplication algorithms, 
as described in the AlphaEvolve paper. Below is the current implementation and its performance.

This is the implementation we're trying to improve:

{current_program}

The current program achieves these metrics:
{metrics}

Areas that need improvement:
{improvement_areas}

Evolution history:
{evolution_history}

Please suggest improvements to make the tensor decomposition algorithm find lower-rank decompositions or
converge faster to valid solutions. Focus on:
- Optimization techniques (loss functions, regularization, schedules)
- Initialization strategies for better convergence
- Methods to encourage sparse, structured, or low-integer coefficient solutions
- Numerical stability improvements

Provide specific code changes using SEARCH/REPLACE blocks as follows:

<<<<<<< SEARCH
// code to be replaced
=======
// new improved code
>>>>>>> REPLACE

Your changes should be well-reasoned and backed by mathematical intuition.
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
