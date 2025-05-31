
from abc import ABC, abstractmethod
import asyncio
from typing import Dict, List, Tuple
from openevolve.utils.async_utils import TaskPool, run_in_executor
from openevolve.config import EvaluatorConfig

class Evaluator(ABC):

    def __init__(self,
                 config: EvaluatorConfig):
        self.config = config
        self.task_pool = TaskPool(max_concurrency=config.parallel_evaluations)

    @abstractmethod
    async def evaluate_program(
        self,
        program_code: str,
        program_id: str = "",
    ) -> Dict[str, float]:
        """
        Evaluate a program and return scores

        Args:
            program_code: Code to evaluate
            program_id: Optional ID for logging

        Returns:
            Dictionary of metric name to score
        """        
        ...

    async def evaluate_multiple(
        self,
        programs: List[Tuple[str, str]],
    ) -> List[Dict[str, float]]:
        """
        Evaluate multiple programs in parallel

        Args:
            programs: List of (program_code, program_id) tuples

        Returns:
            List of metric dictionaries
        """
        tasks = [
            self.task_pool.create_task(self.evaluate_program, program_code, program_id)
            for program_code, program_id in programs
        ]

        return await asyncio.gather(*tasks)
