import asyncio
import os
import uuid
import logging
import time
from dataclasses import dataclass

from openevolve.database import Program, ProgramDatabase
from openevolve.config import Config
from openevolve.evaluator import Evaluator
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.code_utils import (
    apply_diff,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
)


@dataclass
class Result:
    """Resulting program and metrics from an iteration of OpenEvolve"""

    child_program: str = None
    parent: str = None
    child_metrics: str = None
    iteration_time: float = None
    prompt: str = None
    llm_response: str = None
    artifacts: dict = None



def run_iteration_sync(iteration: int, config: Config, evaluation_file: str, database_path: str):
    # setup logger showing PID for current process
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s - %(levelname)s - PID %(process)d - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting iteration in PID %s", os.getpid())

    llm_ensemble = LLMEnsemble(config.llm.models)
    llm_evaluator_ensemble = LLMEnsemble(config.llm.evaluator_models)

    prompt_sampler = PromptSampler(config.prompt)
    evaluator_prompt_sampler = PromptSampler(config.prompt)
    evaluator_prompt_sampler.set_templates("evaluator_system_message")

    # Pass random seed to database if specified
    if config.random_seed is not None:
        config.database.random_seed = config.random_seed

    # load most recent database snapshot
    config.database.db_path = database_path
    database = ProgramDatabase(config.database)

    evaluator = Evaluator(
        config.evaluator,
        evaluation_file,
        llm_evaluator_ensemble,
        evaluator_prompt_sampler,
        database=database,
    )

    # Sample parent and inspirations from current island
    parent, inspirations = database.sample()

    # Get artifacts for the parent program if available
    parent_artifacts = database.get_artifacts(parent.id)

    # Get actual top programs for prompt context (separate from inspirations)
    # This ensures the LLM sees only high-performing programs as examples
    actual_top_programs = database.get_top_programs(5)


    # Build prompt
    prompt = prompt_sampler.build_prompt(
        current_program=parent.code,
        parent_program=parent.code,  # We don't have the parent's code, use the same
        program_metrics=parent.metrics,
        previous_programs=[p.to_dict() for p in database.get_top_programs(3)],
        top_programs=[p.to_dict() for p in actual_top_programs],
        inspirations=[p.to_dict() for p in inspirations],
        language=config.language,
        evolution_round=iteration,
        diff_based_evolution=config.diff_based_evolution,
        program_artifacts=parent_artifacts if parent_artifacts else None,
    )

    async def _run():
        result = Result(parent=parent)
        iteration_start = time.time()

        # Generate code modification
        try:
            llm_response = await llm_ensemble.generate_with_context(
                system_message=prompt["system"],
                messages=[{"role": "user", "content": prompt["user"]}],
            )

            # Parse the response
            if config.diff_based_evolution:
                diff_blocks = extract_diffs(llm_response)

                if not diff_blocks:
                    logger.warning(f"Iteration {iteration+1}: No valid diffs found in response")
                    return

                # Apply the diffs
                child_code = apply_diff(parent.code, llm_response)
                changes_summary = format_diff_summary(diff_blocks)
            else:
                # Parse full rewrite
                new_code = parse_full_rewrite(llm_response, config.language)

                if not new_code:
                    logger.warning(f"Iteration {iteration+1}: No valid code found in response")
                    return

                child_code = new_code
                changes_summary = "Full rewrite"

            # Check code length
            if len(child_code) > config.max_code_length:
                logger.warning(
                    f"Iteration {iteration+1}: Generated code exceeds maximum length "
                    f"({len(child_code)} > {config.max_code_length})"
                )
                return

            # Evaluate the child program
            child_id = str(uuid.uuid4())
            result.child_metrics = await evaluator.evaluate_program(child_code, child_id)
            # Handle artifacts if they exist
            artifacts = evaluator.get_pending_artifacts(child_id)
            # Create a child program
            result.child_program = Program(
                id=child_id,
                code=child_code,
                language=config.language,
                parent_id=parent.id,
                generation=parent.generation + 1,
                metrics=result.child_metrics,
                metadata={
                    "changes": changes_summary,
                    "parent_metrics": parent.metrics,
                },
            )
            result.prompt = prompt
            result.llm_response = llm_response
            # Store artifacts in the result so they can be saved later
            result.artifacts = artifacts

        except Exception as e:
            logger.exception("Error in PID %s:", os.getpid())

        result.iteration_time = time.time() - iteration_start
        return result

    return asyncio.run(_run())
