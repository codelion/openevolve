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
    config.diff_based_evolution = True
    config.allow_full_rewrites = False
    
    # Set database to use performance as the primary metric for comparing programs
    config.database.feature_dimensions = ["performance", "complexity"]
    
    # Create specialized template for matrix multiplication
    from openevolve.prompt.templates import TemplateManager
    
    # Modify prompt templates to use specialized ones for matrix multiplication
    from openevolve.prompt.sampler import PromptSampler
    original_build_prompt = PromptSampler.build_prompt
    
    def custom_build_prompt(self, *args, **kwargs):
        # Get template key from kwargs or use default
        template_key = kwargs.pop('template_key', 'diff_user') if 'template_key' in kwargs else 'diff_user'
        
        # Use specialized template for matrix multiplication
        if template_key == 'diff_user':
            template_key = 'matmul_diff_user'
        
        # Use specialized system message
        if args and len(args) >= 1:
            result = original_build_prompt(self, *args, **kwargs)
            if 'system' in result:
                template_manager = TemplateManager()
                result['system'] = template_manager.get_template('matmul_system')
            return result
        else:
            kwargs['template_key'] = template_key
            return original_build_prompt(self, *args, **kwargs)
    
    # Increase temperature and max_tokens for more creative, complete solutions
    config.llm.temperature = 1.0
    config.llm.max_tokens = 4096
    
    # Configure evaluator to prioritize performance
    config.evaluator.metrics_to_use = ["performance"]
    
    # Initialize OpenEvolve with the custom config
    openevolve = OpenEvolve(
        initial_program_path=str(initial_program_path),
        evaluation_file=str(evaluation_file),
        config=config,  # Pass the config object directly
        config_path=args.config,  # Also pass config_path if provided (lower priority)
        output_dir=str(output_dir),
    )
    
    # Run evolution
    print(f"Starting evolution for {args.iterations} iterations...")
    print(f"Focus on optimizing matrix multiplication for small matrices (2x2 to 5x5)")
    best_program = await openevolve.run(iterations=args.iterations)
    
    print(f"\nEvolution complete!")
    print(f"Best program metrics:")
    for name, value in best_program.metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
