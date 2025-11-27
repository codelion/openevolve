"""
Evaluator for Huawei CodeCraft 2017 CDN optimization problem
"""
import importlib.util
import time
import concurrent.futures
import traceback
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple
from openevolve.evaluation_result import EvaluationResult

# Use a large number instead of infinity for JSON compatibility
INFINITY_COST = 999999999


def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=30):
    """Run a function with timeout"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")


def validate_solution(data: dict, paths: List[Tuple[List[int], int, int, int]]) -> Tuple[bool, str, int]:
    """
    Validate solution correctness

    Returns:
        (is_valid, error_message, actual_cost)
    """
    num_nodes = data['num_nodes']
    edges = data['edges']
    consumers = data['consumers']
    server_types = {tid: (cap, cost) for tid, cap, cost in data['server_types']}
    node_costs = {nid: cost for nid, cost in data['node_deploy_costs']}

    # Build edge capacity map (bidirectional)
    edge_capacity = {}
    edge_costs = {}
    for u, v, bw, cost in edges:
        edge_capacity[(u, v)] = bw
        edge_capacity[(v, u)] = bw
        edge_costs[(u, v)] = cost
        edge_costs[(v, u)] = cost

    # Track deployed servers and used bandwidth
    deployed_servers = {}  # node -> (server_type, used_capacity)
    link_usage = {}  # (u, v) -> used_bandwidth
    consumer_satisfied = {cid: 0 for cid, _, _ in consumers}

    total_server_cost = 0
    total_bandwidth_cost = 0

    # Build consumer map
    consumer_map = {cid: (node, demand) for cid, node, demand in consumers}

    for path, consumer_id, bandwidth, server_type in paths:
        if len(path) == 0:
            return False, f"Empty path for consumer {consumer_id}", 0

        if consumer_id not in consumer_map:
            return False, f"Invalid consumer ID {consumer_id}", 0

        consumer_node, _ = consumer_map[consumer_id]
        server_node = path[0]

        # The path should end at the consumer's connected network node
        # Check if path correctly reaches consumer
        if len(path) == 1:
            # Direct connection: server at same node as consumer
            if path[0] != consumer_node:
                return False, f"Single-node path {path[0]} doesn't match consumer {consumer_id} node {consumer_node}", 0
        else:
            # Multi-hop path: last node should be consumer's connected node
            if path[-1] != consumer_node:
                return False, f"Path end {path[-1]} doesn't match consumer {consumer_id} node {consumer_node}", 0

        # Check server deployment
        if server_node not in deployed_servers:
            if server_type not in server_types:
                return False, f"Invalid server type {server_type}", 0

            capacity, hw_cost = server_types[server_type]
            deploy_cost = node_costs.get(server_node, 0)
            total_server_cost += hw_cost + deploy_cost
            deployed_servers[server_node] = [server_type, 0, capacity]

        # Check server capacity
        _, used_cap, max_cap = deployed_servers[server_node]
        if used_cap + bandwidth > max_cap:
            return False, f"Server at node {server_node} capacity exceeded", 0

        deployed_servers[server_node][1] += bandwidth

        # Check path validity and bandwidth (only for multi-hop paths)
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]

            # Check edge exists
            if (u, v) not in edge_capacity:
                return False, f"Edge ({u}, {v}) does not exist", 0

            # Track bandwidth usage
            if (u, v) not in link_usage:
                link_usage[(u, v)] = 0
            link_usage[(u, v)] += bandwidth

            # Check capacity
            if link_usage[(u, v)] > edge_capacity[(u, v)]:
                return False, f"Edge ({u}, {v}) capacity exceeded: {link_usage[(u, v)]} > {edge_capacity[(u, v)]}", 0

            # Add bandwidth cost
            total_bandwidth_cost += bandwidth * edge_costs[(u, v)]

        # Track consumer satisfaction
        consumer_satisfied[consumer_id] += bandwidth

    # Check all consumers are satisfied
    for consumer_id, connected_node, demand in consumers:
        if consumer_satisfied[consumer_id] < demand:
            return False, f"Consumer {consumer_id} not satisfied: {consumer_satisfied[consumer_id]} < {demand}", 0

    total_cost = total_server_cost + total_bandwidth_cost
    return True, "", total_cost


def load_test_cases(case_dir: str) -> List[Tuple[str, str]]:
    """Load all test cases from directory"""
    case_files = []

    # Search in batch directories
    for batch_dir in glob.glob(os.path.join(case_dir, "batch*")):
        for level_dir in glob.glob(os.path.join(batch_dir, "*")):
            for case_file in glob.glob(os.path.join(level_dir, "case*.txt")):
                case_files.append(case_file)

    return case_files


def evaluate(program_path: str):
    """
    Evaluate the program on multiple test cases

    Args:
        program_path: Path to the program file

    Returns:
        EvaluationResult with metrics
    """
    # Get case directory (relative to program or this evaluator file)
    program_dir = os.path.dirname(os.path.abspath(program_path))
    case_dir = os.path.join(program_dir, "case_example")

    # If case_example doesn't exist relative to program, try relative to this evaluator
    if not os.path.exists(case_dir):
        evaluator_dir = os.path.dirname(os.path.abspath(__file__))
        case_dir = os.path.join(evaluator_dir, "case_example")

    if not os.path.exists(case_dir):
        return EvaluationResult(
            metrics={
                "valid_solutions": 0.0,
                "avg_cost": INFINITY_COST,
                "combined_score": 0.0,
                "error": "Test case directory not found"
            },
            artifacts={
                "error_type": "MissingTestCases",
                "error_message": f"Case directory not found: {case_dir}",
                "suggestion": "Ensure case_example directory exists with test cases"
            }
        )

    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check required function
        if not hasattr(program, "run_solution"):
            return EvaluationResult(
                metrics={
                    "valid_solutions": 0.0,
                    "avg_cost": INFINITY_COST,
                    "combined_score": 0.0,
                    "error": "Missing run_solution function"
                },
                artifacts={
                    "error_type": "MissingFunction",
                    "error_message": "Program missing run_solution function",
                    "suggestion": "Ensure your program has a run_solution(input_text) function"
                }
            )

        # Load test cases (limit to 5 for faster evaluation)
        test_cases = load_test_cases(case_dir)[:5]

        if not test_cases:
            return EvaluationResult(
                metrics={
                    "valid_solutions": 0.0,
                    "avg_cost": INFINITY_COST,
                    "combined_score": 0.0,
                    "error": "No test cases found"
                },
                artifacts={
                    "error_type": "NoTestCases",
                    "error_message": f"No test case files found in {case_dir}",
                    "suggestion": "Verify test case files exist in case_example/batch*/*/case*.txt"
                }
            )

        # Run on test cases
        results = []
        valid_count = 0
        total_time = 0
        error_details = []

        for case_file in test_cases:
            try:
                with open(case_file, 'r', encoding='utf-8') as f:
                    input_text = f.read()

                # Parse input for validation
                data = program.parse_input(input_text)

                start_time = time.time()
                cost, output = run_with_timeout(
                    program.run_solution,
                    args=(input_text,),
                    timeout_seconds=30
                )
                elapsed = time.time() - start_time
                total_time += elapsed

                # Parse output to get paths
                output_lines = [line.strip() for line in output.strip().split('\n') if line.strip()]
                if len(output_lines) < 2:
                    error_details.append(f"{Path(case_file).name}: Invalid output format")
                    continue

                num_paths = int(output_lines[0])
                paths = []

                # Start from line 1 (after the count line), since we filtered out empty lines
                for i in range(1, min(len(output_lines), num_paths + 1)):
                    parts = output_lines[i].split()
                    if len(parts) < 4:  # At least: server_node consumer_id bandwidth server_type
                        continue

                    # Parse path: node1 node2 ... nodeN consumer_id bandwidth server_type
                    # Last 3 elements are: consumer_id, bandwidth, server_type
                    path_nodes = [int(x) for x in parts[:-3]]
                    consumer_id = int(parts[-3])
                    bandwidth = int(parts[-2])
                    server_type = int(parts[-1])

                    paths.append((path_nodes, consumer_id, bandwidth, server_type))

                # Validate solution
                is_valid, error_msg, actual_cost = validate_solution(data, paths)

                if is_valid:
                    valid_count += 1
                    results.append({
                        'case': Path(case_file).name,
                        'cost': actual_cost,
                        'time': elapsed,
                        'valid': True
                    })
                else:
                    error_details.append(f"{Path(case_file).name}: {error_msg}")
                    results.append({
                        'case': Path(case_file).name,
                        'cost': INFINITY_COST,
                        'time': elapsed,
                        'valid': False,
                        'error': error_msg
                    })

            except TimeoutError as e:
                error_details.append(f"{Path(case_file).name}: Timeout")
                results.append({'case': Path(case_file).name, 'cost': INFINITY_COST, 'valid': False, 'error': 'Timeout'})
            except Exception as e:
                error_details.append(f"{Path(case_file).name}: {str(e)}")
                results.append({'case': Path(case_file).name, 'cost': INFINITY_COST, 'valid': False, 'error': str(e)})

        # Calculate metrics
        if valid_count == 0:
            return EvaluationResult(
                metrics={
                    "valid_solutions": 0.0,
                    "avg_cost": INFINITY_COST,
                    "combined_score": 0.0,
                    "error": "All test cases failed"
                },
                artifacts={
                    "error_type": "AllCasesFailed",
                    "error_message": f"All {len(test_cases)} test cases failed",
                    "error_details": error_details[:5],
                    "suggestion": "Check algorithm correctness and constraint handling"
                }
            )

        valid_costs = [r['cost'] for r in results if r['valid']]
        avg_cost = sum(valid_costs) / len(valid_costs)
        success_rate = valid_count / len(test_cases)
        avg_time = total_time / len(test_cases)

        # Normalize cost to score (lower cost = higher score)
        # Use reference cost of 10000 as baseline
        cost_score = 10000.0 / (avg_cost + 1.0)

        # Time score (faster is better)
        # Exponential decay: 1.0 at 0s, ~0.5 at 5s, ~0.1 at 15s, approaches 0 for longer times
        # This heavily penalizes slow solutions since contest limits are 30s-90s
        import math
        time_score = math.exp(-avg_time / 10.0)  # Decay factor of 10 seconds

        # For very fast solutions (< 1s), give bonus
        if avg_time < 1.0:
            time_score = min(1.0, time_score * 1.2)

        # Combined score: 40% success rate + 35% cost + 25% time
        # Increased time weight from 10% to 25% since it's a critical ranking factor
        combined_score = 0.4 * success_rate + 0.35 * min(1.0, cost_score) + 0.25 * time_score

        return EvaluationResult(
            metrics={
                "valid_solutions": success_rate,
                "avg_cost": avg_cost,
                "avg_time": avg_time,
                "cost_score": cost_score,
                "time_score": time_score,
                "combined_score": combined_score,
            },
            artifacts={
                "num_test_cases": len(test_cases),
                "num_valid": valid_count,
                "results": results[:3],  # First 3 results
                "avg_time_per_case": f"{avg_time:.2f}s",
                "total_time": f"{total_time:.2f}s",
                "time_breakdown": {
                    "fastest": f"{min(r['time'] for r in results if r['valid']):.2f}s" if valid_count > 0 else "N/A",
                    "slowest": f"{max(r['time'] for r in results if r['valid']):.2f}s" if valid_count > 0 else "N/A",
                },
                "scoring_weights": {
                    "success_rate": "40%",
                    "cost": "35%",
                    "time": "25%"
                }
            }
        )

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        print(traceback.format_exc())

        return EvaluationResult(
            metrics={
                "valid_solutions": 0.0,
                "avg_cost": INFINITY_COST,
                "combined_score": 0.0,
                "error": str(e)
            },
            artifacts={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "full_traceback": traceback.format_exc(),
                "suggestion": "Check for syntax errors or import issues"
            }
        )


def evaluate_stage1(program_path: str):
    """Stage 1: Quick validation with single test case"""
    program_dir = os.path.dirname(os.path.abspath(program_path))
    case_dir = os.path.join(program_dir, "case_example")

    # If case_example doesn't exist relative to program, try relative to this evaluator
    if not os.path.exists(case_dir):
        evaluator_dir = os.path.dirname(os.path.abspath(__file__))
        case_dir = os.path.join(evaluator_dir, "case_example")

    try:
        # Load program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        if not hasattr(program, "run_solution"):
            return EvaluationResult(
                metrics={"runs_successfully": 0.0, "combined_score": 0.0},
                artifacts={"error": "Missing run_solution function"}
            )

        # Load one test case
        test_cases = load_test_cases(case_dir)[:1]
        if not test_cases:
            return EvaluationResult(
                metrics={"runs_successfully": 0.0, "combined_score": 0.0},
                artifacts={"error": "No test cases found"}
            )

        with open(test_cases[0], 'r', encoding='utf-8') as f:
            input_text = f.read()

        # Try to run
        cost, output = run_with_timeout(program.run_solution, args=(input_text,), timeout_seconds=10)

        # Basic validation
        if cost > 0 and len(output) > 0:
            return EvaluationResult(
                metrics={"runs_successfully": 1.0, "combined_score": 0.5},
                artifacts={"stage1_cost": cost}
            )
        else:
            return EvaluationResult(
                metrics={"runs_successfully": 0.5, "combined_score": 0.25},
                artifacts={"warning": "Solution returned but may be invalid"}
            )

    except TimeoutError:
        return EvaluationResult(
            metrics={"runs_successfully": 0.0, "combined_score": 0.0},
            artifacts={"error": "Timeout"}
        )
    except Exception as e:
        return EvaluationResult(
            metrics={"runs_successfully": 0.0, "combined_score": 0.0},
            artifacts={"error": str(e)}
        )


def evaluate_stage2(program_path: str):
    """Stage 2: Full evaluation"""
    return evaluate(program_path)
