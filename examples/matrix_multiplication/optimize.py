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
Your task is to optimize a tensor decomposition algorithm for discovering efficient matrix multiplication.

When matrix multiplication is viewed as a tensor problem, the goal is to find a minimum-rank decomposition
of the corresponding 3D tensor. Each term in the decomposition corresponds to a scalar multiplication in 
the algorithm, so minimizing the rank directly leads to faster matrix multiplication.

Your focus should be on targeted, specific improvements to the tensor decomposition optimization process. 
Make small, focused changes rather than rewriting large sections. Key areas to consider include:

1. Loss function improvements (regularization terms that encourage specific properties)
2. Adding noise injection or annealing schedules to avoid local minima
3. Improving initialization strategies for better convergence
4. Numerical stability enhancements for complex-valued operations
5. Techniques to encourage integer or half-integer coefficients

The best algorithms from the literature for various matrix sizes include:
- 2x2 matrices: Strassen's algorithm (7 multiplications)
- 3x3 matrices: Laderman's algorithm (23 multiplications) 
- 4x4 matrices: Recursive Strassen (49 multiplications, improved to 48 with complex values)

Focus on making 1-3 specific, high-impact changes rather than comprehensive rewrites.
"""
    
    # User message template for tensor decomposition optimization (shortened for compatibility)
    user_template = """
Improve the tensor decomposition algorithm below with 1-3 specific changes.

CURRENT CODE:
{current_program}

METRICS:
{metrics}

FOCUS AREAS:
{improvement_areas}

Make targeted changes using this format:

<<<<<<< SEARCH
// exact code to match (keep short)
=======
// improved code
>>>>>>> REPLACE

RULES:
1. Each SEARCH block must exactly match existing code
2. Focus on 1-3 specific changes only
3. Explain each change briefly
4. Avoid large rewrites
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
