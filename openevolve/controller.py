"""
Main controller for OpenEvolve
"""
import asyncio
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from openevolve.config import Config, load_config
from openevolve.database import Program, ProgramDatabase
from openevolve.evaluator import Evaluator
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.code_utils import (
    apply_diff,
    extract_code_language,
    extract_diffs,
    format_diff_summary,
    parse_evolve_blocks,
    parse_full_rewrite,
)

logger = logging.getLogger(__name__)


class OpenEvolve:
    """
    Main controller for OpenEvolve
    
    Orchestrates the evolution process, coordinating between the prompt sampler,
    LLM ensemble, evaluator, and program database.
    
    Features:
    - Tracks the absolute best program across evolution steps
    - Ensures the best solution is not lost during the MAP-Elites process
    - Always includes the best program in the selection process for inspiration
    - Maintains detailed logs and metadata about improvements
    """
    
    def __init__(
        self,
        initial_program_path: str,
        evaluation_file: str,
        config_path: Optional[str] = None,
        config: Optional[Config] = None,
        output_dir: Optional[str] = None,
    ):
        # Load configuration
        if config is not None:
            # Use provided Config object directly
            self.config = config
        else:
            # Load from file or use defaults
            self.config = load_config(config_path)
        
        # Set up output directory
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(initial_program_path), "openevolve_output"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Load initial program
        self.initial_program_path = initial_program_path
        self.initial_program_code = self._load_initial_program()
        self.language = extract_code_language(self.initial_program_code)
        
        # Extract file extension from initial program
        self.file_extension = os.path.splitext(initial_program_path)[1]
        if not self.file_extension:
            # Default to .py if no extension found
            self.file_extension = ".py"
        else:
            # Make sure it starts with a dot
            if not self.file_extension.startswith("."):
                self.file_extension = f".{self.file_extension}"
        
        # Initialize components
        self.llm_ensemble = LLMEnsemble(self.config.llm)
        self.prompt_sampler = PromptSampler(self.config.prompt)
        self.database = ProgramDatabase(self.config.database)
        self.evaluator = Evaluator(
            self.config.evaluator, 
            evaluation_file, 
            self.llm_ensemble
        )
        
        logger.info(
            f"Initialized OpenEvolve with {initial_program_path} "
            f"and {evaluation_file}"
        )
    
    def _setup_logging(self) -> None:
        """Set up logging"""
        log_dir = self.config.log_dir or os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level))
        
        # Add file handler
        log_file = os.path.join(log_dir, f"openevolve_{time.strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        root_logger.addHandler(file_handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        root_logger.addHandler(console_handler)
        
        logger.info(f"Logging to {log_file}")
    
    def _load_initial_program(self) -> str:
        """Load the initial program from file"""
        with open(self.initial_program_path, "r") as f:
            return f.read()
    
    async def run(
        self,
        iterations: Optional[int] = None,
        target_score: Optional[float] = None,
    ) -> Program:
        """
        Run the evolution process
        
        Args:
            iterations: Maximum number of iterations (uses config if None)
            target_score: Target score to reach (continues until reached if specified)
            
        Returns:
            Best program found
        """
        max_iterations = iterations or self.config.max_iterations
        
        # Initialize the database with the initial program
        initial_program_id = str(uuid.uuid4())
        
        # Evaluate the initial program
        initial_metrics = await self.evaluator.evaluate_program(
            self.initial_program_code, initial_program_id
        )
        
        initial_program = Program(
            id=initial_program_id,
            code=self.initial_program_code,
            language=self.language,
            metrics=initial_metrics,
        )
        
        self.database.add(initial_program)
        
        # Main evolution loop
        start_iteration = self.database.last_iteration
        total_iterations = start_iteration + max_iterations
        
        logger.info(f"Starting evolution from iteration {start_iteration} for {max_iterations} iterations (total: {total_iterations})")
        
        for i in range(start_iteration, total_iterations):
            iteration_start = time.time()
            
            # Sample parent and inspirations
            parent, inspirations = self.database.sample()
            
            # Build prompt
            prompt = self.prompt_sampler.build_prompt(
                current_program=parent.code,
                parent_program=parent.code,  # We don't have the parent's code, use the same
                program_metrics=parent.metrics,
                previous_programs=[p.to_dict() for p in self.database.get_top_programs(3)],
                top_programs=[p.to_dict() for p in inspirations],
                language=self.language,
                evolution_round=i,
                allow_full_rewrite=self.config.allow_full_rewrites,
            )
            
            # Generate code modification
            try:
                llm_response = await self.llm_ensemble.generate_with_context(
                    system_message=prompt["system"],
                    messages=[{"role": "user", "content": prompt["user"]}],
                )
                
                # Parse the response
                if self.config.diff_based_evolution:
                    diff_blocks = extract_diffs(llm_response)
                    
                    if not diff_blocks:
                        logger.warning(f"Iteration {i+1}: No valid diffs found in response")
                        continue
                    
                    # Apply the diffs
                    child_code = apply_diff(parent.code, llm_response)
                    changes_summary = format_diff_summary(diff_blocks)
                else:
                    # Parse full rewrite
                    new_code = parse_full_rewrite(llm_response, self.language)
                    
                    if not new_code:
                        logger.warning(f"Iteration {i+1}: No valid code found in response")
                        continue
                    
                    child_code = new_code
                    changes_summary = "Full rewrite"
                
                # Check code length
                if len(child_code) > self.config.max_code_length:
                    logger.warning(
                        f"Iteration {i+1}: Generated code exceeds maximum length "
                        f"({len(child_code)} > {self.config.max_code_length})"
                    )
                    continue
                
                # Evaluate the child program
                child_id = str(uuid.uuid4())
                child_metrics = await self.evaluator.evaluate_program(
                    child_code, child_id
                )
                
                # Create a child program
                child_program = Program(
                    id=child_id,
                    code=child_code,
                    language=self.language,
                    parent_id=parent.id,
                    generation=parent.generation + 1,
                    metrics=child_metrics,
                    metadata={
                        "changes": changes_summary,
                        "parent_metrics": parent.metrics,
                    },
                )
                
                # Add to database
                self.database.add(child_program)
                
                # Log progress
                iteration_time = time.time() - iteration_start
                self._log_iteration(i, parent, child_program, iteration_time)
                
                # Specifically check if this is the new best program
                if self.database.best_program_id == child_program.id:
                    logger.info(f"🌟 New best solution found at iteration {i+1}: {child_program.id}")
                    logger.info(f"Metrics: {', '.join(f'{name}={value:.4f}' for name, value in child_program.metrics.items())}")
                
                # Save checkpoint
                if (i + 1) % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(i + 1)
                
                # Check if target score reached
                if target_score is not None:
                    avg_score = sum(child_metrics.values()) / max(1, len(child_metrics))
                    if avg_score >= target_score:
                        logger.info(
                            f"Target score {target_score} reached after {i+1} iterations"
                        )
                        break
            
            except Exception as e:
                logger.error(f"Error in iteration {i+1}: {str(e)}")
                continue
        
        # Get the best program using our tracking mechanism
        best_program = None
        if self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
            logger.info(f"Using tracked best program: {self.database.best_program_id}")
        
        # Fallback to calculating best program if tracked program not found
        if best_program is None:
            best_program = self.database.get_best_program()
            logger.info("Using calculated best program (tracked program not found)")
            
        # Check if there's a better program by combined_score that wasn't tracked
        if "combined_score" in best_program.metrics:
            best_by_combined = self.database.get_best_program(metric="combined_score")
            if best_by_combined and best_by_combined.id != best_program.id and "combined_score" in best_by_combined.metrics:
                # If the combined_score of this program is significantly better, use it instead
                if best_by_combined.metrics["combined_score"] > best_program.metrics["combined_score"] + 0.02:
                    logger.warning(f"Found program with better combined_score: {best_by_combined.id}")
                    logger.warning(f"Score difference: {best_program.metrics['combined_score']:.4f} vs {best_by_combined.metrics['combined_score']:.4f}")
                    best_program = best_by_combined
        
        if best_program:
            logger.info(
                f"Evolution complete. Best program has metrics: "
                f"{', '.join(f'{name}={value:.4f}' for name, value in best_program.metrics.items())}"
            )
            
            # Save the best program (using our tracked best program)
            self._save_best_program()
            
            return best_program
        else:
            logger.warning("No valid programs found during evolution")
            return initial_program
    
    def _log_iteration(
        self,
        iteration: int,
        parent: Program,
        child: Program,
        elapsed_time: float,
    ) -> None:
        """
        Log iteration progress
        
        Args:
            iteration: Iteration number
            parent: Parent program
            child: Child program
            elapsed_time: Elapsed time in seconds
        """
        # Calculate improvement
        improvement = {}
        for metric, value in child.metrics.items():
            if metric in parent.metrics:
                diff = value - parent.metrics[metric]
                improvement[metric] = diff
        
        improvement_str = ", ".join(
            f"{name}={diff:+.4f}" for name, diff in improvement.items()
        )
        
        logger.info(
            f"Iteration {iteration+1}: Child {child.id} from parent {parent.id} "
            f"in {elapsed_time:.2f}s. Metrics: "
            f"{', '.join(f'{name}={value:.4f}' for name, value in child.metrics.items())} "
            f"(Δ: {improvement_str})"
        )
    
    def _save_checkpoint(self, iteration: int) -> None:
        """
        Save a checkpoint
        
        Args:
            iteration: Current iteration number
        """
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save the database
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}")
        self.database.save(checkpoint_path, iteration)
        
        logger.info(f"Saved checkpoint at iteration {iteration} to {checkpoint_path}")
    
    def _save_best_program(self, program: Optional[Program] = None) -> None:
        """
        Save the best program
        
        Args:
            program: Best program (if None, uses the tracked best program)
        """
        # If no program is provided, use the tracked best program from the database
        if program is None:
            if self.database.best_program_id:
                program = self.database.get(self.database.best_program_id)
            else:
                # Fallback to calculating best program if no tracked best program
                program = self.database.get_best_program()
                
        if not program:
            logger.warning("No best program found to save")
            return
            
        best_dir = os.path.join(self.output_dir, "best")
        os.makedirs(best_dir, exist_ok=True)
        
        # Use the extension from the initial program file
        filename = f"best_program{self.file_extension}"
        code_path = os.path.join(best_dir, filename)
        
        with open(code_path, "w") as f:
            f.write(program.code)
        
        # Save complete program info including metrics
        info_path = os.path.join(best_dir, "best_program_info.json")
        with open(info_path, "w") as f:
            import json
            json.dump({
                "id": program.id,
                "generation": program.generation,
                "timestamp": program.timestamp,
                "parent_id": program.parent_id,
                "metrics": program.metrics,
                "language": program.language,
                "saved_at": time.time()
            }, f, indent=2)
        
        logger.info(f"Saved best program to {code_path} with program info to {info_path}")
