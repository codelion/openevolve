"""
Evaluator for blocksworld planner.
Tests the planner on various problems and returns performance metrics.
"""
import importlib.util
import time
import random
from typing import List, Dict, Optional
from openevolve.evaluation_result import EvaluationResult


def load_program(program_path: str):
    """Load the program to evaluate."""
    spec = importlib.util.spec_from_file_location("program", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_valid_state(state: Dict[str, str], blocks: List[str]) -> bool:
    """
    Check if a blocksworld state is valid (no cycles, each block appears once).

    Args:
        state: Dictionary mapping blocks to their locations
        blocks: List of all block names

    Returns:
        True if state is valid, False otherwise
    """
    # Check all blocks are present
    if set(state.keys()) != set(blocks):
        return False

    # Check no block appears as a location more than once (except 'table')
    locations = [loc for loc in state.values() if loc != 'table']
    if len(locations) != len(set(locations)):
        return False  # Duplicate location (two blocks on same block)

    # Check for cycles by following the chain from each block
    for start_block in blocks:
        visited = set()
        current = start_block

        while current != 'table':
            if current in visited:
                return False  # Cycle detected
            visited.add(current)
            current = state.get(current, 'table')

            # Safety check: if we visit more blocks than exist, something is wrong
            if len(visited) > len(blocks):
                return False

    return True


def generate_random_stack_problem(n_blocks: int, seed: Optional[int] = None) -> Dict:
    """
    Generate a random stacking problem.
    All blocks start on table, goal is a random single tower.

    Args:
        n_blocks: Number of blocks to use
        seed: Random seed for reproducibility

    Returns:
        Problem dictionary with blocks, initial, goal, and optimal_length
    """
    if seed is not None:
        random.seed(seed)

    # Generate block names
    blocks = [chr(65 + i) for i in range(n_blocks)]  # A, B, C, ...

    # Initial: all on table
    initial = {block: 'table' for block in blocks}

    # Goal: create a random tower from bottom to top
    shuffled = blocks.copy()
    random.shuffle(shuffled)

    goal = {}
    # Build tower bottom-up: bottom block on table, each next block on previous
    goal[shuffled[0]] = 'table'  # bottom block
    for i in range(1, len(shuffled)):
        goal[shuffled[i]] = shuffled[i - 1]  # each block stacks on the one below

    # Verify states are valid
    assert is_valid_state(initial, blocks), "Generated invalid initial state"
    assert is_valid_state(goal, blocks), "Generated invalid goal state"

    # Optimal is 2 * (n_blocks - 1) for simple stacking from table
    optimal_length = 2 * (n_blocks - 1)

    return {
        'blocks': blocks,
        'initial': initial,
        'goal': goal,
        'optimal_length': optimal_length
    }


def generate_random_rearrange_problem(n_blocks: int, seed: Optional[int] = None) -> Dict:
    """
    Generate a harder random rearrangement problem.
    Creates random initial stacks and random goal configuration.

    Args:
        n_blocks: Number of blocks to use
        seed: Random seed for reproducibility

    Returns:
        Problem dictionary with blocks, initial, goal, and estimated optimal_length
    """
    if seed is not None:
        random.seed(seed)

    # Generate block names
    blocks = [chr(65 + i) for i in range(n_blocks)]  # A, B, C, ...

    def build_random_stacks(block_list: List[str]) -> Dict[str, str]:
        """Build random stacks from a list of blocks."""
        state = {}
        remaining = block_list.copy()
        random.shuffle(remaining)

        while remaining:
            # Start a new stack with 1-4 blocks
            stack_size = min(random.randint(1, 4), len(remaining))
            stack = [remaining.pop() for _ in range(stack_size)]

            # Build the stack from bottom to top
            # Bottom block on table, each subsequent block on the previous one
            state[stack[0]] = 'table'
            for i in range(1, len(stack)):
                state[stack[i]] = stack[i - 1]

        return state

    # Create random initial configuration (multiple stacks)
    initial = build_random_stacks(blocks)

    # Create random goal configuration
    goal = build_random_stacks(blocks)

    # Verify states are valid
    assert is_valid_state(initial, blocks), "Generated invalid initial state"
    assert is_valid_state(goal, blocks), "Generated invalid goal state"

    # Estimate optimal - harder to compute exactly, use heuristic
    # Count blocks that are in wrong position
    misplaced = sum(1 for b in blocks if initial.get(b) != goal.get(b))
    optimal_length = max(2 * misplaced, n_blocks)  # rough estimate

    return {
        'blocks': blocks,
        'initial': initial,
        'goal': goal,
        'optimal_length': optimal_length
    }


def generate_problem_suite(block_counts: List[int], problems_per_size: int = 2,
                          seed: Optional[int] = None) -> List[Dict]:
    """
    Generate a suite of problems of varying sizes.

    Args:
        block_counts: List of block counts to generate problems for
        problems_per_size: Number of problems to generate per size
        seed: Base random seed for reproducibility

    Returns:
        List of problem dictionaries
    """
    problems = []

    for block_count in block_counts:
        for i in range(problems_per_size):
            problem_seed = None if seed is None else seed + block_count * 100 + i

            # Alternate between stack and rearrange problems
            if i % 2 == 0:
                problem = generate_random_stack_problem(block_count, problem_seed)
                problem['type'] = 'stack'
            else:
                problem = generate_random_rearrange_problem(block_count, problem_seed)
                problem['type'] = 'rearrange'

            problems.append(problem)

    return problems


def evaluate_program(program_path: str, timeout_seconds: int = 60,
                    use_random: bool = False, random_sizes: Optional[List[int]] = None,
                    random_seed: int = 42) -> dict:
    """
    Evaluate the blocksworld planner on test problems.

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum time per problem in seconds (default: 60)
        use_random: If True, use randomly generated problems instead of hardcoded ones
        random_sizes: List of block counts for random problems (default: [6, 8, 10, 12, 15])
        random_seed: Seed for random problem generation (default: 42)

    Returns:
        Dictionary with metrics for the planner's performance
    """
    try:
        program = load_program(program_path)
    except Exception as e:
        return {
            'combined_score': 0.0,
            'success_rate': 0.0,
            'avg_plan_length': 100.0,
            'error': str(e)
        }
    
    # Test problems of increasing difficulty
    if use_random:
        # Generate random problems
        if random_sizes is None:
            random_sizes = [6, 8, 10, 12, 15]
        # Use 1 problem per size to keep evaluations fast (avoid timeout)
        test_problems = generate_problem_suite(random_sizes, problems_per_size=1, seed=random_seed)
    else:
        # Use hardcoded baseline problems
        test_problems = [
            # Simple: stack 2 blocks
            {
                'blocks': ['A', 'B'],
                'initial': {'A': 'table', 'B': 'table'},
                'goal': {'A': 'B', 'B': 'table'},
                'optimal_length': 2  # pickup A, stack A on B
            },
            # Medium: stack 3 blocks
            {
                'blocks': ['A', 'B', 'C'],
                'initial': {'A': 'table', 'B': 'table', 'C': 'table'},
                'goal': {'A': 'B', 'B': 'C', 'C': 'table'},
                'optimal_length': 4  # pickup B, stack B on C, pickup A, stack A on B
            },
            # Medium: restack 3 blocks
            {
                'blocks': ['A', 'B', 'C'],
                'initial': {'A': 'B', 'B': 'C', 'C': 'table'},
                'goal': {'C': 'B', 'B': 'A', 'A': 'table'},
                'optimal_length': 6  # unstacking and restacking
            },
            # Harder: 4 blocks
            {
                'blocks': ['A', 'B', 'C', 'D'],
                'initial': {'A': 'table', 'B': 'table', 'C': 'table', 'D': 'table'},
                'goal': {'A': 'B', 'B': 'C', 'C': 'D', 'D': 'table'},
                'optimal_length': 6
            },
            # Complex: rearrange 4 blocks
            {
                'blocks': ['A', 'B', 'C', 'D'],
                'initial': {'A': 'B', 'B': 'table', 'C': 'D', 'D': 'table'},
                'goal': {'B': 'A', 'A': 'table', 'D': 'C', 'C': 'table'},
                'optimal_length': 8
            },
            # Very hard: 5 blocks
            {
                'blocks': ['A', 'B', 'C', 'D', 'E'],
                'initial': {'A': 'table', 'B': 'table', 'C': 'table', 'D': 'table', 'E': 'table'},
                'goal': {'A': 'B', 'B': 'C', 'C': 'D', 'D': 'E', 'E': 'table'},
                'optimal_length': 8
            },
        ]
    
    successes = 0
    total_problems = len(test_problems)
    plan_lengths = []
    efficiency_scores = []
    solve_times = []
    timeouts_count = 0

    for i, problem in enumerate(test_problems):
        print(f"\n{'='*60}")
        problem_type = problem.get('type', 'baseline')
        print(f"Problem {i+1}: {len(problem['blocks'])} blocks ({problem_type})")
        print(f"Initial: {problem['initial']}")
        print(f"Goal:    {problem['goal']}")
        print(f"Optimal length: {problem['optimal_length']}")

        solve_time = 0.0
        timed_out = False

        try:
            # Track time to solve
            start_time = time.perf_counter()

            # Call solve_problem with timeout for this specific problem
            result = None
            def solve_wrapper():
                nonlocal result
                result = program.solve_problem(
                    problem['blocks'],
                    problem['initial'],
                    problem['goal']
                )

            import threading
            thread = threading.Thread(target=solve_wrapper)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_seconds)

            solve_time = time.perf_counter() - start_time

            if thread.is_alive():
                # Timeout occurred
                timed_out = True
                timeouts_count += 1
                plan_lengths.append(100)
                efficiency_scores.append(0.0)
                solve_times.append(timeout_seconds)
                print(f"⏱ TIMEOUT - Exceeded {timeout_seconds}s limit")
            elif result and result['success']:
                successes += 1
                plan_length = result['plan_length']
                plan_lengths.append(plan_length)
                solve_times.append(solve_time)

                # Efficiency: how close to optimal
                optimal = problem['optimal_length']
                if plan_length > 0:
                    efficiency = min(1.0, optimal / plan_length)
                    efficiency_scores.append(efficiency)
                else:
                    efficiency_scores.append(0.0)

                # Show only first 3 actions to avoid clutter
                plan_preview = result['plan'][:3] if len(result['plan']) > 3 else result['plan']
                plan_str = f"{plan_preview}..." if len(result['plan']) > 3 else str(plan_preview)

                print(f"✓ SUCCESS - Plan length: {plan_length} (efficiency: {efficiency:.2%}) - Time: {solve_time:.3f}s")
                print(f"Plan (first 3): {plan_str}")
            else:
                plan_lengths.append(100)  # penalty for failure
                efficiency_scores.append(0.0)
                solve_times.append(solve_time)
                print(f"✗ FAILED - No solution found (Time: {solve_time:.3f}s)")

        except Exception as e:
            # Problem solving failed
            solve_time = time.perf_counter() - start_time if 'start_time' in locals() else 0
            print(f"✗ ERROR: {e} (Time: {solve_time:.3f}s)")
            plan_lengths.append(100)
            efficiency_scores.append(0.0)
            solve_times.append(0)
    
    print(f"\n{'='*60}")

    # Calculate metrics
    success_rate = successes / total_problems
    avg_plan_length = sum(plan_lengths) / len(plan_lengths) if plan_lengths else 100.0
    avg_efficiency = sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0.0
    avg_solve_time = sum(solve_times) / len(solve_times) if solve_times else timeout_seconds

    # Combined score: balance success rate and efficiency
    # Success rate is more important, so weight it higher
    combined_score = 0.7 * success_rate + 0.3 * avg_efficiency

    return {
        'combined_score': combined_score,
        'success_rate': success_rate,
        'avg_plan_length': avg_plan_length,
        'avg_efficiency': avg_efficiency,
        'avg_solve_time': avg_solve_time,
        'timeouts': timeouts_count,
    }


def evaluate(program_path: str) -> EvaluationResult:
    """
    Main evaluation function for OpenEvolve compatibility.
    Tests problems sizes 6-100 with early termination on first failure.

    Args:
        program_path: Path to the program file

    Returns:
        EvaluationResult with progress_score, efficiency_score, combined_score
    """
    # Test all problem sizes 6-100 (95 problems total)
    # Early termination on first failure
    problem_sizes = list(range(6, 101))

    return evaluate_problem_set(
        program_path=program_path,
        problem_sizes=problem_sizes,
        timeout_per_problem=5,
        seed=42,
        stage_name="Evaluation"
    )


def evaluate_problem_set(program_path: str, problem_sizes: List[int],
                         timeout_per_problem: int, seed: int = 42,
                         stage_name: str = "") -> EvaluationResult:
    """
    Evaluate a program on a set of problems with early termination.

    Args:
        program_path: Path to the program file
        problem_sizes: List of block counts to test (e.g., [6, 7, 8])
        timeout_per_problem: Timeout in seconds for each problem
        seed: Random seed for problem generation
        stage_name: Name of the stage for logging/artifacts

    Returns:
        EvaluationResult with progress_score, efficiency_score, combined_score
    """
    try:
        program = load_program(program_path)
    except Exception as e:
        return EvaluationResult(
            metrics={
                'progress_score': 0.0,
                'efficiency_score': 0.0,
                'combined_score': 0.0,
                'error': str(e)
            },
            artifacts={
                'stage': stage_name,
                'error_type': 'LoadError',
                'error_message': str(e)
            }
        )

    # Generate one problem per size (alternating stack/rearrange)
    problems = []
    for i, size in enumerate(problem_sizes):
        problem_seed = seed + size * 100
        if i % 2 == 0:
            problem = generate_random_stack_problem(size, problem_seed)
            problem['type'] = 'stack'
        else:
            problem = generate_random_rearrange_problem(size, problem_seed)
            problem['type'] = 'rearrange'
        problems.append(problem)

    problems_solved = 0
    efficiency_scores = []

    for i, problem in enumerate(problems):
        problem_type = problem.get('type', 'unknown')
        size = len(problem['blocks'])

        try:
            # Track time to solve
            start_time = time.perf_counter()

            # Call solve_problem with timeout
            result = None
            def solve_wrapper():
                nonlocal result
                result = program.solve_problem(
                    problem['blocks'],
                    problem['initial'],
                    problem['goal']
                )

            import threading
            thread = threading.Thread(target=solve_wrapper)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_per_problem)

            solve_time = time.perf_counter() - start_time

            if thread.is_alive():
                # Timeout - stop cascade immediately
                print(f"{stage_name} - Problem {i+1} (size {size}, {problem_type}): TIMEOUT after {timeout_per_problem}s")
                break
            elif result and result['success']:
                # Success!
                problems_solved += 1
                plan_length = result['plan_length']
                optimal = problem['optimal_length']

                # Calculate efficiency for this problem
                if plan_length > 0 and optimal > 0:
                    efficiency = min(1.0, optimal / plan_length)
                    efficiency_scores.append(efficiency)

                print(f"{stage_name} - Problem {i+1} (size {size}, {problem_type}): SUCCESS - "
                      f"Plan length: {plan_length} (optimal: {optimal}) - Time: {solve_time:.3f}s")
            else:
                # Failure - stop cascade immediately
                print(f"{stage_name} - Problem {i+1} (size {size}, {problem_type}): FAILED - No solution found")
                break

        except Exception as e:
            # Error - stop cascade immediately
            print(f"{stage_name} - Problem {i+1} (size {size}, {problem_type}): ERROR - {e}")
            break

    # Calculate metrics (normalized 0-1)
    # Progress is relative to ALL 95 problems (sizes 6-100)
    TOTAL_PROBLEMS = 95
    progress_score = problems_solved / TOTAL_PROBLEMS

    # Efficiency is average of successful problems
    efficiency_score = sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0.0

    # Combined score: 66% progress, 34% efficiency
    combined_score = 0.66 * progress_score + 0.34 * efficiency_score

    print(f"{stage_name} Results: Solved {problems_solved}/{len(problems)} problems | "
          f"Progress: {progress_score:.4f} | Efficiency: {efficiency_score:.4f} | Combined: {combined_score:.4f}")

    return EvaluationResult(
        metrics={
            'progress_score': progress_score,
            'efficiency_score': efficiency_score,
            'combined_score': combined_score,
            'problems_solved': problems_solved,
        },
        artifacts={
            'stage': stage_name,
            'problems_attempted': len(problems),
            'problems_solved': problems_solved,
            'max_size_tested': problem_sizes[min(problems_solved, len(problems) - 1)],
        }
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate blocksworld planner')
    parser.add_argument('program_path', help='Path to the planner program')
    parser.add_argument('timeout', nargs='?', type=int, default=60,
                       help='Timeout in seconds per problem (default: 60)')
    parser.add_argument('--random', action='store_true',
                       help='Use randomly generated problems instead of hardcoded baseline')
    parser.add_argument('--sizes', type=int, nargs='+',
                       help='Block counts for random problems (default: 6 8 10 12 15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for problem generation (default: 42)')

    args = parser.parse_args()

    if args.random:
        if args.sizes:
            print(f"Evaluating with random problems (sizes: {args.sizes}, seed: {args.seed})")
        else:
            print(f"Evaluating with random problems (default sizes: 6, 8, 10, 12, 15, seed: {args.seed})")
    else:
        print("Evaluating with baseline hardcoded problems")

    print(f"Timeout: {args.timeout} seconds per problem...")

    metrics = evaluate_program(
        args.program_path,
        timeout_seconds=args.timeout,
        use_random=args.random,
        random_sizes=args.sizes,
        random_seed=args.seed
    )

    print("\nEvaluation Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")