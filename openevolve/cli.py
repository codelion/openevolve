"""
Command-line interface for OpenEvolve
"""
import argparse
import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional

from openevolve import OpenEvolve
from openevolve.config import Config, load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="OpenEvolve - Evolutionary coding agent")
    
    parser.add_argument(
        "initial_program",
        help="Path to the initial program file"
    )
    
    parser.add_argument(
        "evaluation_file",
        help="Path to the evaluation file containing an 'evaluate' function"
    )
    
    parser.add_argument(
        "--config",
        "-c",
        help="Path to configuration file (YAML)",
        default=None
    )
    
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory for results",
        default=None
    )
    
    parser.add_argument(
        "--iterations",
        "-i",
        help="Maximum number of iterations",
        type=int,
        default=None
    )
    
    parser.add_argument(
        "--target-score",
        "-t",
        help="Target score to reach",
        type=float,
        default=None
    )
    
    parser.add_argument(
        "--log-level",
        "-l",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO"
    )
    
    return parser.parse_args()


async def main_async() -> int:
    """
    Main asynchronous entry point
    
    Returns:
        Exit code
    """
    args = parse_args()
    
    # Check if files exist
    if not os.path.exists(args.initial_program):
        print(f"Error: Initial program file '{args.initial_program}' not found")
        return 1
    
    if not os.path.exists(args.evaluation_file):
        print(f"Error: Evaluation file '{args.evaluation_file}' not found")
        return 1
    
    # Initialize OpenEvolve
    try:
        openevolve = OpenEvolve(
            initial_program_path=args.initial_program,
            evaluation_file=args.evaluation_file,
            config_path=args.config,
            output_dir=args.output,
        )
        
        # Override log level if specified
        if args.log_level:
            logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # Run evolution
        best_program = await openevolve.run(
            iterations=args.iterations,
            target_score=args.target_score,
        )
        
        print(f"\nEvolution complete!")
        print(f"Best program metrics:")
        for name, value in best_program.metrics.items():
            print(f"  {name}: {value:.4f}")
        
        return 0
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def main() -> int:
    """
    Main entry point
    
    Returns:
        Exit code
    """
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
