"""
Improved parallel processing using threads with shared memory
"""

import asyncio
import logging
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Dict, List, Optional

from openevolve.config import Config
from openevolve.database import ProgramDatabase
from openevolve.evaluator import Evaluator
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.prompt.sampler import PromptSampler
from openevolve.iteration import run_iteration_with_shared_db

logger = logging.getLogger(__name__)


class ThreadedEvaluationPool:
    """
    Thread-based parallel evaluation pool for improved performance
    
    Uses threads instead of processes to avoid pickling issues while
    still providing parallelism for I/O-bound LLM calls.
    """
    
    def __init__(self, config: Config, evaluation_file: str, database: ProgramDatabase):
        self.config = config
        self.evaluation_file = evaluation_file
        self.database = database
        
        self.num_workers = config.evaluator.parallel_evaluations
        self.executor = None
        
        # Pre-initialize components for each thread
        self.thread_local = threading.local()
        
        logger.info(f"Initializing threaded evaluation pool with {self.num_workers} workers")
    
    def start(self) -> None:
        """Start the thread pool"""
        self.executor = ThreadPoolExecutor(
            max_workers=self.num_workers, 
            thread_name_prefix="EvalWorker"
        )
        logger.info(f"Started thread pool with {self.num_workers} threads")
    
    def stop(self) -> None:
        """Stop the thread pool"""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        logger.info("Stopped thread pool")
    
    def submit_evaluation(self, iteration: int) -> Future:
        """
        Submit an evaluation task to the thread pool
        
        Args:
            iteration: Iteration number to evaluate
            
        Returns:
            Future that will contain the result
        """
        if not self.executor:
            raise RuntimeError("Thread pool not started")
        
        return self.executor.submit(self._run_evaluation, iteration)
    
    def _run_evaluation(self, iteration: int):
        """Run evaluation in a worker thread"""
        # Get or create thread-local components
        if not hasattr(self.thread_local, 'initialized'):
            self._initialize_thread_components()
        
        try:
            # Run the iteration
            result = asyncio.run(run_iteration_with_shared_db(
                iteration,
                self.config,
                self.database,  # Shared database (thread-safe reads)
                self.thread_local.evaluator,
                self.thread_local.llm_ensemble,
                self.thread_local.prompt_sampler
            ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error in thread evaluation {iteration}: {e}")
            return None
    
    def _initialize_thread_components(self) -> None:
        """Initialize components for this thread"""
        thread_id = threading.get_ident()
        logger.debug(f"Initializing components for thread {thread_id}")
        
        try:
            # Initialize LLM components
            self.thread_local.llm_ensemble = LLMEnsemble(self.config.llm.models)
            self.thread_local.llm_evaluator_ensemble = LLMEnsemble(self.config.llm.evaluator_models)
            
            # Initialize prompt samplers
            self.thread_local.prompt_sampler = PromptSampler(self.config.prompt)
            self.thread_local.evaluator_prompt_sampler = PromptSampler(self.config.prompt)
            self.thread_local.evaluator_prompt_sampler.set_templates("evaluator_system_message")
            
            # Initialize evaluator
            self.thread_local.evaluator = Evaluator(
                self.config.evaluator,
                self.evaluation_file,
                self.thread_local.llm_evaluator_ensemble,
                self.thread_local.evaluator_prompt_sampler,
                database=self.database,
            )
            
            self.thread_local.initialized = True
            logger.debug(f"Initialized components for thread {thread_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize thread components: {e}")
            raise


class ImprovedParallelController:
    """
    Controller for improved parallel processing using shared memory and threads
    """
    
    def __init__(self, config: Config, evaluation_file: str, database: ProgramDatabase):
        self.config = config
        self.evaluation_file = evaluation_file
        self.database = database
        
        self.thread_pool = None
        self.database_lock = threading.RLock()  # For database writes
        self.shutdown_flag = threading.Event()  # For graceful shutdown
        
    def start(self) -> None:
        """Start the improved parallel system"""
        self.thread_pool = ThreadedEvaluationPool(
            self.config, self.evaluation_file, self.database
        )
        self.thread_pool.start()
        
        logger.info("Started improved parallel controller")
    
    def stop(self) -> None:
        """Stop the improved parallel system"""
        self.shutdown_flag.set()  # Signal shutdown
        
        if self.thread_pool:
            self.thread_pool.stop()
            self.thread_pool = None
        
        logger.info("Stopped improved parallel controller")
    
    def request_shutdown(self) -> None:
        """Request graceful shutdown (for signal handlers)"""
        logger.info("Graceful shutdown requested...")
        self.shutdown_flag.set()
    
    async def run_evolution(
        self, start_iteration: int, max_iterations: int, target_score: Optional[float] = None,
        checkpoint_callback=None
    ):
        """
        Run evolution with improved parallel processing
        
        Args:
            start_iteration: Starting iteration number
            max_iterations: Maximum number of iterations
            target_score: Target score to achieve
            
        Returns:
            Best program found
        """
        total_iterations = start_iteration + max_iterations
        
        logger.info(
            f"Starting improved parallel evolution from iteration {start_iteration} "
            f"for {max_iterations} iterations (total: {total_iterations})"
        )
        
        # Submit initial batch of evaluations
        pending_futures = {}
        batch_size = min(self.config.evaluator.parallel_evaluations * 2, max_iterations)
        
        for i in range(start_iteration, min(start_iteration + batch_size, total_iterations)):
            future = self.thread_pool.submit_evaluation(i)
            pending_futures[i] = future
        
        next_iteration_to_submit = start_iteration + batch_size
        completed_iterations = 0
        
        # Island management
        programs_per_island = max(1, max_iterations // (self.config.database.num_islands * 10))
        current_island_counter = 0
        
        # Process results as they complete
        while pending_futures and completed_iterations < max_iterations and not self.shutdown_flag.is_set():
            # Find completed futures
            completed_iteration = None
            for iteration, future in list(pending_futures.items()):
                if future.done():
                    completed_iteration = iteration
                    break
            
            if completed_iteration is None:
                # No results ready, wait a bit
                await asyncio.sleep(0.01)
                continue
            
            # Process completed result
            future = pending_futures.pop(completed_iteration)
            
            try:
                result = future.result()
                
                if result and hasattr(result, 'child_program') and result.child_program:
                    # Thread-safe database update
                    with self.database_lock:
                        self.database.add(result.child_program, iteration=completed_iteration)
                        
                        # Store artifacts if they exist
                        if result.artifacts:
                            self.database.store_artifacts(result.child_program.id, result.artifacts)
                        
                        # Log prompts
                        if hasattr(result, 'prompt') and result.prompt:
                            self.database.log_prompt(
                                template_key=(
                                    "full_rewrite_user" if not self.config.diff_based_evolution 
                                    else "diff_user"
                                ),
                                program_id=result.child_program.id,
                                prompt=result.prompt,
                                responses=[result.llm_response] if hasattr(result, 'llm_response') else [],
                            )
                        
                        # Manage island evolution
                        if completed_iteration > start_iteration and current_island_counter >= programs_per_island:
                            self.database.next_island()
                            current_island_counter = 0
                            logger.debug(f"Switched to island {self.database.current_island}")
                        
                        current_island_counter += 1
                        
                        # Increment generation for current island
                        self.database.increment_island_generation()
                        
                        # Check migration
                        if self.database.should_migrate():
                            logger.info(f"Performing migration at iteration {completed_iteration}")
                            self.database.migrate_programs()
                            self.database.log_island_status()
                    
                    # Log progress (outside lock)
                    logger.info(
                        f"Iteration {completed_iteration}: "
                        f"Program {result.child_program.id} "
                        f"(parent: {result.parent.id if result.parent else 'None'}) "
                        f"completed in {result.iteration_time:.2f}s"
                    )
                    
                    if result.child_program.metrics:
                        metrics_str = ", ".join([
                            f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                            for k, v in result.child_program.metrics.items()
                        ])
                        logger.info(f"Metrics: {metrics_str}")
                    
                    # Check for new best program
                    if self.database.best_program_id == result.child_program.id:
                        logger.info(
                            f"🌟 New best solution found at iteration {completed_iteration}: "
                            f"{result.child_program.id}"
                        )
                    
                    # Save checkpoints at intervals
                    if completed_iteration % self.config.checkpoint_interval == 0:
                        logger.info(f"Checkpoint interval reached at iteration {completed_iteration}")
                        self.database.log_island_status()
                        if checkpoint_callback:
                            checkpoint_callback(completed_iteration)
                    
                    # Check target score
                    if target_score is not None and result.child_program.metrics:
                        numeric_metrics = [
                            v for v in result.child_program.metrics.values() 
                            if isinstance(v, (int, float))
                        ]
                        if numeric_metrics:
                            avg_score = sum(numeric_metrics) / len(numeric_metrics)
                            if avg_score >= target_score:
                                logger.info(
                                    f"Target score {target_score} reached after {completed_iteration} iterations"
                                )
                                break
                else:
                    logger.warning(f"No valid result from iteration {completed_iteration}")
                
            except Exception as e:
                logger.error(f"Error processing result from iteration {completed_iteration}: {e}")
            
            completed_iterations += 1
            
            # Submit next iteration if available
            if next_iteration_to_submit < total_iterations:
                future = self.thread_pool.submit_evaluation(next_iteration_to_submit)
                pending_futures[next_iteration_to_submit] = future
                next_iteration_to_submit += 1
        
        # Handle shutdown or completion
        if self.shutdown_flag.is_set():
            logger.info("Shutdown requested, canceling remaining evaluations...")
            # Cancel remaining futures
            for iteration, future in pending_futures.items():
                future.cancel()
                logger.debug(f"Canceled iteration {iteration}")
        else:
            # Wait for any remaining futures if not shutting down
            for iteration, future in pending_futures.items():
                try:
                    future.result(timeout=10.0)
                except Exception as e:
                    logger.warning(f"Error waiting for iteration {iteration}: {e}")
        
        if self.shutdown_flag.is_set():
            logger.info("Evolution interrupted by shutdown")
        else:
            logger.info("Evolution completed")