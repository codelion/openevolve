#!/usr/bin/env python
"""
Circle Packing with Hierarchical Evolution

This example demonstrates the hierarchical abstraction layer system on the
circle packing problem (n=26). The goal is to replicate or improve over
AlphaEvolve's result of 2.635.

The hierarchical system will:
1. Start with L1 (code details) evolution
2. Escalate to L2 (implementation patterns) when L1 plateaus
3. Escalate to L3 (architectural components) when L2 plateaus
4. Use different model tiers for different layers
5. Build rich context from the Evolutionary Memory Graph
6. Extract insights periodically to guide evolution

Usage:
    python run_hierarchical.py [--iterations N] [--checkpoint PATH]
"""

import asyncio
import argparse
import logging
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openevolve import OpenEvolve
from openevolve.config import load_config


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Circle Packing with Hierarchical Evolution")
    parser.add_argument(
        "--iterations", type=int, default=300, help="Number of iterations to run"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint directory to resume from"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file path (default: config.yaml)",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Get paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    initial_program_path = os.path.join(current_dir, "initial_program.py")
    evaluator_path = os.path.join(current_dir, "evaluator.py")
    config_path = os.path.join(current_dir, args.config)

    # Check if files exist
    if not os.path.exists(initial_program_path):
        logger.error(f"Initial program not found: {initial_program_path}")
        return

    if not os.path.exists(evaluator_path):
        logger.error(f"Evaluator not found: {evaluator_path}")
        return

    if not os.path.exists(config_path):
        logger.error(f"Config not found: {config_path}")
        return

    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # Verify hierarchical evolution is enabled
    if not config.hierarchical.enabled:
        logger.warning("Hierarchical evolution is DISABLED in config!")
        logger.warning("To enable, set hierarchical.enabled: true in config.yaml")
    else:
        logger.info("✅ Hierarchical evolution is ENABLED")
        logger.info(f"  - Model tiers configured: {len(config.hierarchical.tier0_models) > 0}")
        logger.info(f"  - EMG enabled: {config.hierarchical.emg_enabled}")
        logger.info(f"  - Insight extraction: {config.hierarchical.enable_insight_extraction}")

    # Create OpenEvolve instance
    logger.info("Initializing OpenEvolve with hierarchical evolution...")
    openevolve = OpenEvolve(
        initial_program_path=initial_program_path,
        evaluation_file=evaluator_path,
        config=config,
    )

    # Run evolution
    logger.info(f"Starting evolution for {args.iterations} iterations")
    logger.info(f"Target: Replicate AlphaEvolve's result of 2.635 for n=26")
    logger.info("-" * 80)

    try:
        best_program = await openevolve.run(
            iterations=args.iterations,
            checkpoint_path=args.checkpoint,
        )

        if best_program:
            logger.info("-" * 80)
            logger.info("🎉 Evolution complete!")
            logger.info(f"Best score: {best_program.metrics.get('combined_score', 0):.6f}")
            logger.info(f"Sum of radii: {best_program.metrics.get('sum_radii', 0):.6f}")
            logger.info(f"Target (AlphaEvolve): 2.635")
            logger.info(
                f"Ratio: {best_program.metrics.get('sum_radii', 0) / 2.635 * 100:.2f}%"
            )
            logger.info(f"Generation found: {best_program.generation}")
            logger.info(f"Iteration found: {best_program.iteration_found}")

            # Print hierarchical statistics if available
            if hasattr(openevolve, "hierarchical_orchestrator"):
                stats = openevolve.hierarchical_orchestrator.get_statistics()
                logger.info("\n📊 Hierarchical Evolution Statistics:")
                logger.info(f"  Current generation: {stats.get('current_generation', 0)}")

                if "transition_manager" in stats:
                    tm_stats = stats["transition_manager"]
                    logger.info("\n  Layer Status:")
                    for layer, status in tm_stats.get("layer_status", {}).items():
                        logger.info(
                            f"    {layer}: best={status['best_score']:.4f}, "
                            f"attempts={status['attempts']}, "
                            f"success_rate={status['success_rate']:.2%}"
                        )

                if "emg" in stats:
                    emg_stats = stats["emg"]
                    logger.info(f"\n  EMG: {emg_stats['total_nodes']} nodes, {emg_stats['total_edges']} edges")

                if "insights_count" in stats:
                    logger.info(f"  Insights extracted: {stats['insights_count']}")

        else:
            logger.warning("No best program found")

    except KeyboardInterrupt:
        logger.info("\n⚠️  Evolution interrupted by user")
    except Exception as e:
        logger.error(f"❌ Evolution failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
