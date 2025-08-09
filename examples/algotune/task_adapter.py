#!/usr/bin/env python3
"""
Task adapter for converting AlgoTune tasks to OpenEvolve format.

This adapter extracts AlgoTune tasks from an external repository and converts them to OpenEvolve format,
creating the necessary initial_program.py, evaluator.py, and config.yaml files.
"""

import os
import sys
import importlib.util
import shutil
import ast
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

class AlgoTuneTaskAdapter:
    """Adapter to convert AlgoTune tasks to OpenEvolve format."""
    
    def __init__(self, algotune_path: Optional[str] = None, task: Optional[str] = None):
        """
        Initialize the adapter.
        
        Args:
            algotune_path: Path to AlgoTune repository directory (e.g., /path/to/AlgoTune)
            task: Task name to create OpenEvolve files for
        """
        if algotune_path is None:
            raise ValueError("Please specify algotune_path to the AlgoTune repository directory.")
        
        self.algotune_path = Path(algotune_path)
        self.algotune_tasks_path = self.algotune_path / "AlgoTuneTasks"
        self.algotuner_path = self.algotune_path / "AlgoTuner"
        self.output_path = Path(__file__).parent
        self.task = task
        # Validate paths exist
        if not self.algotune_tasks_path.exists():
            raise ValueError(f"AlgoTuneTasks directory not found at: {self.algotune_tasks_path}")
        if not self.algotuner_path.exists():
            raise ValueError(f"AlgoTuner directory not found at: {self.algotuner_path}")
        
        # Add AlgoTune paths to Python path for importing
        self._setup_import_paths()
        
        # Load all available tasks
        self._load_tasks()

        if self.task is not None:
            if self.task not in self.available_tasks:
                raise ValueError(f"Task '{self.task}' not found. Available tasks: {list(self.available_tasks.keys())}")
            self.task_info = self.available_tasks[self.task]
            self.task_name = self.task  # Use the task name directly
    
    def _setup_import_paths(self):
        """Setup Python import paths for AlgoTune modules."""
        # Add AlgoTune base directory to path
        if str(self.algotune_path) not in sys.path:
            sys.path.insert(0, str(self.algotune_path))
        
        # Try to import AlgoTune modules
        try:
            from AlgoTuneTasks.base import TASK_REGISTRY
            from AlgoTuneTasks.registry import TASK_REGISTRY as REGISTRY_TASK_REGISTRY
            print(f"Successfully imported AlgoTune modules from {self.algotune_path}")
        except ImportError as e:
            print(f"Warning: Could not import AlgoTune tasks: {e}")
            print(f"Make sure AlgoTune is properly installed and accessible")
            print(f"AlgoTune path: {self.algotune_path}")
            TASK_REGISTRY = {}
            REGISTRY_TASK_REGISTRY = {}
    
    def _load_tasks(self):
        """Load all available AlgoTune tasks."""
        self.available_tasks = {}
        
        # Scan the tasks directory
        for task_dir in self.algotune_tasks_path.iterdir():
            if task_dir.is_dir() and not task_dir.name.startswith('_'):
                task_name = task_dir.name
                description_file = task_dir / "description.txt"
                task_file = task_dir / f"{task_name}.py"
                
                if description_file.exists() and task_file.exists():
                    self.available_tasks[task_name] = {
                        'path': task_dir,
                        'description_file': description_file,
                        'task_file': task_file
                    }
        
        print(f"Loaded {len(self.available_tasks)} tasks from {self.algotune_tasks_path}")
    
    def get_task_description(self, task_name: str) -> str:
        """Get the description of a task."""
        if task_name not in self.available_tasks:
            raise ValueError(f"Task '{task_name}' not found. Available tasks: {list(self.available_tasks.keys())}")
        
        description_file = self.available_tasks[task_name]['description_file']
        with open(description_file, 'r') as f:
            return f.read().strip()
    
    def _extract_task_class_info(self, task_name: str) -> Dict[str, Any]:
        """Extract class information from the task file with improved method extraction."""
        task_info = self.available_tasks[task_name]
        
        # Read the task file
        with open(task_info['task_file'], 'r') as f:
            task_code = f.read()
        
        # Parse the AST to find the class
        try:
            tree = ast.parse(task_code)
        except Exception as e:
            print(f"Error parsing AST for {task_name}: {e}")
            raise
        
        class_info = {
            'name': None,
            'solve_method': None,
            'generate_problem_method': None,
            'is_solution_method': None,
            'imports': [],
            'class_code': None
        }
        
        # Extract imports with improved filtering
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_str = f"import {alias.name}"
                    if alias.asname:
                        import_str += f" as {alias.asname}"
                    
                    # Filter out AlgoTune-specific imports
                    if not any(x in import_str for x in ['AlgoTune', 'register_task', 'Task']):
                        class_info['imports'].append(import_str)
                        
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    import_str = f"from {module} import {alias.name}"
                    if alias.asname:
                        import_str += f" as {alias.asname}"
                    
                    # Filter out AlgoTune-specific imports
                    if not any(x in import_str for x in ['AlgoTune', 'register_task', 'Task']):
                        class_info['imports'].append(import_str)
        
        # Find the task class and extract the solve method
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if this class inherits from Task or has the task name
                is_task_class = False
                if node.bases is not None:
                    for base in node.bases:
                        base_str = ast.unparse(base) if hasattr(ast, 'unparse') else str(base)
                        if 'Task' in base_str:
                            is_task_class = True
                            break
                
                # Also check if the class name matches the task name (case-insensitive)
                if not is_task_class and task_name.lower() in node.name.lower():
                    is_task_class = True
                
                if is_task_class:
                    class_info['name'] = node.name
                    
                    # Extract the entire class code
                    class_info['class_code'] = ast.unparse(node)
                    
                    # Find the solve method using AST
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == 'solve':
                            try:
                                # Get the source lines for this method
                                method_start = item.lineno - 1  # Convert to 0-based index
                                method_end = item.end_lineno if hasattr(item, 'end_lineno') else method_start + 1
                                # Extract the method source code
                                source_lines = task_code.split('\n')
                                method_source_lines = source_lines[method_start:method_end]
                                # Extract the method body with proper indentation
                                body_lines = []
                                def_indent = len(method_source_lines[0]) - len(method_source_lines[0].lstrip())
                                signature_end = 0
                                for i, line in enumerate(method_source_lines):
                                    if ':' in line and line.strip().endswith(':'):
                                        signature_end = i
                                        break
                                for line in method_source_lines[signature_end + 1:]:
                                    if line.strip():
                                        line_indent = len(line) - len(line.lstrip())
                                        if line_indent > def_indent:
                                            dedented_line = line[def_indent:]
                                            body_lines.append('            ' + dedented_line)
                                        elif line_indent == def_indent and line.strip().startswith('def '):
                                            break
                                        elif line_indent == def_indent:
                                            break
                                    else:
                                        body_lines.append('')
                                if body_lines:
                                    min_indent = float('inf')
                                    for line in body_lines:
                                        if line.strip():
                                            indent = len(line) - len(line.lstrip())
                                            min_indent = min(min_indent, indent)
                                    if min_indent != float('inf'):
                                        fixed_lines = []
                                        for line in body_lines:
                                            if line.strip():
                                                current_indent = len(line) - len(line.lstrip())
                                                relative_indent = current_indent - min_indent
                                                additional_spaces = relative_indent
                                                new_indent = '            ' + (' ' * additional_spaces)
                                                stripped = line.strip()
                                                fixed_lines.append(new_indent + stripped)
                                            else:
                                                fixed_lines.append('')
                                        body_lines = fixed_lines
                                if body_lines:
                                    class_info['solve_method'] = '\n'.join(body_lines)
                                else:
                                    class_info['solve_method'] = '            # Placeholder for solve method\n            pass'
                            except Exception as e:
                                class_info['solve_method'] = '            # Placeholder for solve method\n            pass'
                            break
                    break
        
        return class_info
    
    def _generate_initial_program(self, task_name: str) -> str:
        """Generate the initial program for OpenEvolve based on the actual task implementation."""
        task_info = self.available_tasks[task_name]
        description = self.get_task_description(task_name)
        class_info = self._extract_task_class_info(task_name)
        
        if not class_info['name']:
            raise ValueError(f"Could not find Task class in {task_name}")
        
        # Create imports section - remove duplicates and filter problematic imports
        unique_imports = []
        seen_imports = set()
        
        # Filter out AlgoTune-specific imports that won't be available
        problematic_imports = [
            'from AlgoTuneTasks.base import',
            'import AlgoTuneTasks',
            'from AlgoTuneTasks.',
            'import AlgoTuneTasks.'
        ]
        
        for imp in class_info['imports']:
            # Skip problematic imports
            if any(problematic in imp for problematic in problematic_imports):
                continue
                
            if imp not in seen_imports:
                unique_imports.append(imp)
                seen_imports.add(imp)
        
        # Add essential imports for OpenEvolve environment
        essential_imports = [
            'import logging',
            'import numpy as np',
            'from typing import Any, Dict, List, Optional'
        ]
        
        # Remove duplicate typing imports
        unique_imports = [imp for imp in unique_imports if not imp.startswith('from typing import')]
        
        for imp in essential_imports:
            if imp not in seen_imports:
                unique_imports.append(imp)
                seen_imports.add(imp)
        
        imports = "\n".join(unique_imports)
        

        
        # Use the actual solve method from the original task
        solve_method = class_info['solve_method']
        if solve_method:
            # The method body is already properly indented from extraction
            method_body = solve_method
        else:
            # Fallback to task-specific method if extraction failed
            method_body = self._generate_task_specific_method(task_name, solve_method, class_info)
        
        # Clean the description for use in docstring
        import re
        docstring_description = description.replace('\\', '\\\\')
        # Use simple string replacement instead of regex for better reliability
        docstring_description = docstring_description.replace('\\x', '\\\\x')
        docstring_description = docstring_description.replace('b\\x', 'b\\\\x')
        
        # Additional fixes for problematic byte literals
        # Replace byte literals with safer representations
        docstring_description = re.sub(r'b\\\\x[0-9a-fA-F]{2}', 'b\'\\\\x00\'', docstring_description)
        docstring_description = re.sub(r'\\\\x[0-9a-fA-F]{2}', '\\\\x00', docstring_description)
        
        # Fix any remaining problematic patterns
        docstring_description = docstring_description.replace('\\\\xencrypted', '\\\\x00encrypted')
        docstring_description = docstring_description.replace('\\\\xauthentication', '\\\\x00authentication')
        
        initial_program = f'''# EVOLVE-BLOCK-START
"""
{docstring_description}

This is the initial implementation that will be evolved by OpenEvolve.
The solve method will be improved through evolution.
"""
{imports}

class {class_info['name']}:
    """
    Initial implementation of {task_name} task.
    This will be evolved by OpenEvolve to improve performance and correctness.
    """
    
    def __init__(self):
        """Initialize the {class_info['name']}."""
        pass
    
    def solve(self, problem):
        """
        Solve the {task_name} problem.
        
        Args:
            problem: Dictionary containing problem data specific to {task_name}
                   
        Returns:
            The solution in the format expected by the task
        """
        try:
{method_body}
            
        except Exception as e:
            logging.error(f"Error in solve method: {{e}}")
            raise e

def run_solver(problem):
    """
    Main function to run the solver.
    This function is used by the evaluator to test the evolved solution.
    
    Args:
        problem: The problem to solve
        
    Returns:
        The solution
    """
    solver = {class_info['name']}()
    return solver.solve(problem)

# EVOLVE-BLOCK-END

# Test function for evaluation
if __name__ == "__main__":
    # Example usage
    print("Initial {task_name} implementation ready for evolution")
'''
        
        return initial_program
    
    def _generate_evaluator(self, task_name: str) -> str:
        """Generate the evaluator for OpenEvolve using the actual task implementation with baseline comparison."""
        task_info = self.available_tasks[task_name]
        description = self.get_task_description(task_name)
        class_info = self._extract_task_class_info(task_name)
        
        evaluator = f'''"""
Evaluator for the {task_name} task with baseline comparison

This evaluator compares OpenEvolve's evolved solutions against the reference
AlgoTune baseline implementation to measure performance improvements.
The speedup becomes the primary fitness score for evolution.
"""

import importlib.util
import numpy as np
import time
import concurrent.futures
import traceback
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Add AlgoTune to path for importing reference tasks
# These paths will be dynamically determined based on the AlgoTune installation
# The adapter will handle path setup when the evaluator is created

# Setup AlgoTune paths dynamically
def setup_algotune_paths():
    """Setup Python import paths for AlgoTune modules."""
    # The AlgoTune path should be passed as a parameter to the evaluator
    possible_algotune_paths = [
        Path(__file__).parent.parent.parent.parent / "AlgoTune",
        Path.home() / "github" / "AlgoTune",
    ]
    
    algotune_base = None
    for path in possible_algotune_paths:
        if path.exists():
            algotune_base = path
            break
    
    if algotune_base is None:
        print("Warning: Could not find AlgoTune installation")
        return False
    
    # Add AlgoTune base directory to path
    if str(algotune_base) not in sys.path:
        sys.path.insert(0, str(algotune_base))
    
    return True

# Setup paths and try to import AlgoTune tasks
if setup_algotune_paths():
    try:
        from AlgoTuneTasks.base import TASK_REGISTRY
        # Import the specific {task_name} task to register it
        from AlgoTuneTasks.{task_name}.{task_name} import {class_info['name']}
        print("Successfully imported AlgoTune tasks and {task_name}")
    except ImportError as e:
        print(f"Error: Could not import AlgoTune tasks: {{e}}")
        print("Make sure AlgoTune is properly installed and accessible")
        TASK_REGISTRY = {{}}
else:
    print("Warning: Could not setup AlgoTune paths")
    TASK_REGISTRY = {{}}

def run_with_timeout(func, args=(), kwargs={{}}, timeout_seconds=30):
    """
    Run a function with a timeout using concurrent.futures

    Args:
        func: Function to run
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        timeout_seconds: Timeout in seconds

    Returns:
        Result of the function or raises TimeoutError
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {{timeout_seconds}} seconds")

def safe_convert(value):
    """Convert a value safely for evaluation"""
    try:
        if isinstance(value, (list, tuple)):
            return [safe_convert(v) for v in value]
        elif isinstance(value, np.ndarray):
            return value.tolist()
        else:
            return value
    except Exception:
        return value

def calculate_speedup(baseline_time_ms: float, evolved_time_ms: float, is_valid: bool) -> Optional[float]:
    """
    Calculate speedup between baseline and evolved solution.
    
    Speedup = (Baseline Time) / (Evolved Time)
    Higher is better.
    
    Args:
        baseline_time_ms: Time taken by baseline implementation
        evolved_time_ms: Time taken by evolved solution
        is_valid: Whether the evolved solution is valid
        
    Returns:
        Speedup value or None if calculation is not possible
    """
    if not is_valid:
        return None
        
    if baseline_time_ms is None or baseline_time_ms <= 0:
        return None
        
    if evolved_time_ms is None:
        return None
        
    if evolved_time_ms <= 0:
        return float('inf')  # Infinite speedup for instant solution
        
    return baseline_time_ms / evolved_time_ms

def measure_baseline_performance(task_instance, problem, num_runs=3, warmup_runs=1):
    """
    Measure baseline performance using the original AlgoTune implementation.
    
    Args:
        task_instance: The AlgoTune task instance
        problem: Problem to solve
        num_runs: Number of timing runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Dictionary with baseline timing results
    """
    try:
        # Warmup runs
        for _ in range(warmup_runs):
            try:
                task_instance.solve(problem)
            except Exception:
                pass  # Ignore warmup errors
                
        # Timing runs
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            try:
                result = task_instance.solve(problem)
                end_time = time.perf_counter()
                if result is not None:
                    elapsed_ms = (end_time - start_time) * 1000
                    times.append(elapsed_ms)
            except Exception as e:
                print(f"Baseline run failed: {{e}}")
                continue
                
        if not times:
            return {{
                "success": False,
                "error": "All baseline runs failed",
                "avg_time_ms": None,
                "min_time_ms": None,
                "std_time_ms": None
            }}
            
        return {{
            "success": True,
            "avg_time_ms": float(np.mean(times)),
            "min_time_ms": float(np.min(times)),
            "std_time_ms": float(np.std(times)),
            "times": times
        }}
        
    except Exception as e:
        return {{
            "success": False,
            "error": str(e),
            "avg_time_ms": None,
            "min_time_ms": None,
            "std_time_ms": None
        }}

def measure_evolved_performance(program, problem, num_runs=3, warmup_runs=1, timeout_seconds=30):
    """
    Measure evolved solution performance.
    
    Args:
        program: The evolved program module
        problem: Problem to solve
        num_runs: Number of timing runs
        warmup_runs: Number of warmup runs
        timeout_seconds: Timeout per run
        
    Returns:
        Dictionary with evolved timing results
    """
    try:
        # Warmup runs
        for _ in range(warmup_runs):
            try:
                run_with_timeout(program.run_solver, args=(problem,), timeout_seconds=timeout_seconds)
            except Exception:
                pass  # Ignore warmup errors
                
        # Timing runs
        times = []
        results = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            try:
                result = run_with_timeout(program.run_solver, args=(problem,), timeout_seconds=timeout_seconds)
                end_time = time.perf_counter()
                elapsed_ms = (end_time - start_time) * 1000
                times.append(elapsed_ms)
                results.append(result)
            except TimeoutError:
                print(f"Evolved solution timed out after {{timeout_seconds}} seconds")
                continue
            except Exception as e:
                print(f"Evolved run failed: {{e}}")
                continue
                
        if not times:
            return {{
                "success": False,
                "error": "All evolved runs failed",
                "avg_time_ms": None,
                "min_time_ms": None,
                "std_time_ms": None,
                "results": []
            }}
            
        return {{
            "success": True,
            "avg_time_ms": float(np.mean(times)),
            "min_time_ms": float(np.min(times)),
            "std_time_ms": float(np.std(times)),
            "times": times,
            "results": results
        }}
        
    except Exception as e:
        return {{
            "success": False,
            "error": str(e),
            "avg_time_ms": None,
            "min_time_ms": None,
            "std_time_ms": None,
            "results": []
        }}

def evaluate(program_path, config=None):
    """
    Enhanced evaluation with baseline comparison for {task_name} task.
    
    This evaluator:
    1. Loads the evolved solve method from initial_program.py
    2. Generates test problems using the original AlgoTune task
    3. Measures baseline performance using original AlgoTune implementation
    4. Measures evolved solution performance
    5. Calculates speedup as primary fitness score
    6. Validates correctness using the original task's validation method

    Args:
        program_path: Path to the evolved program file (initial_program.py)
        config: Configuration dictionary with evaluator settings

    Returns:
        Dictionary of metrics including speedup as primary fitness score
    """
    try:
        # Load configuration
        if config is None:
            config = {{
                "algotune": {{
                    "num_trials": 5,
                    "data_size": 5,
                    "timeout": 30,
                    "num_runs": 3,
                    "warmup_runs": 1
                }}
            }}
        
        # Extract AlgoTune task-specific settings from config
        algotune_config = config.get("algotune", {{}})
        num_trials = algotune_config.get("num_trials", 5)
        data_size = algotune_config.get("data_size", 5)
        timeout_seconds = algotune_config.get("timeout", 30)
        num_runs = algotune_config.get("num_runs", 3)
        warmup_runs = algotune_config.get("warmup_runs", 1)
        
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if the required function exists
        if not hasattr(program, "run_solver"):
            print(f"Error: program does not have 'run_solver' function")
            return {{
                "correctness_score": 0.0,
                "performance_score": 0.0,
                "combined_score": 0.0,
                "speedup_score": 0.0,  # Primary fitness score
                "baseline_comparison": {{
                    "mean_speedup": None,
                    "median_speedup": None,
                    "success_rate": 0.0,
                    "baseline_times": [],
                    "evolved_times": [],
                    "speedups": []
                }},
                "error": "Missing run_solver function",
            }}

        # Get the original task for reference solutions and problem generation
        task_class = None
        if "{task_name}" in TASK_REGISTRY:
            task_class = TASK_REGISTRY["{task_name}"]
            print(f"Successfully loaded {task_name} task from registry")
        else:
            print(f"Error: {task_name} task not found in TASK_REGISTRY")
            print(f"Available tasks: {{list(TASK_REGISTRY.keys())}}")
            raise Exception("Could not load {task_name} task from AlgoTune registry")

        # Generate test problems and evaluate
        correctness_scores = []
        performance_scores = []
        baseline_times = []
        evolved_times = []
        speedups = []
        valid_count = 0
        success_count = 0

        for trial in range(num_trials):
            try:
                # Generate a test problem using the original task
                if task_class:
                    task_instance = task_class()
                    problem = task_instance.generate_problem(n=data_size, random_seed=trial)
                else:
                    raise Exception("Could not load original AlgoTune task for problem generation")

                # Measure baseline performance
                baseline_result = measure_baseline_performance(
                    task_instance, problem, num_runs, warmup_runs
                )
                
                if not baseline_result["success"]:
                    print(f"Trial {{trial}}: Baseline measurement failed: {{baseline_result.get('error', 'Unknown error')}}")
                    continue

                # Measure evolved performance
                evolved_result = measure_evolved_performance(
                    program, problem, num_runs, warmup_runs, timeout_seconds
                )
                
                if not evolved_result["success"]:
                    print(f"Trial {{trial}}: Evolved measurement failed: {{evolved_result.get('error', 'Unknown error')}}")
                    continue

                # Validate evolved solution
                correctness_score = 0.0
                is_valid = False
                
                if evolved_result["results"]:
                    # Use the first result for validation
                    evolved_solution = evolved_result["results"][0]
                    evolved_solution = safe_convert(evolved_solution)
                    
                    try:
                        is_valid = task_instance.is_solution(problem, evolved_solution)
                        correctness_score = 1.0 if is_valid else 0.0
                    except Exception as e:
                        print(f"Trial {{trial}}: Error checking solution validity: {{e}}")
                        correctness_score = 0.0
                        is_valid = False

                # Calculate speedup
                baseline_time = baseline_result["min_time_ms"]  # Use minimum time for fair comparison
                evolved_time = evolved_result["min_time_ms"]
                speedup = calculate_speedup(baseline_time, evolved_time, is_valid)

                # Store results
                correctness_scores.append(correctness_score)
                baseline_times.append(baseline_time)
                evolved_times.append(evolved_time)
                
                if speedup is not None:
                    speedups.append(speedup)
                    valid_count += 1
                
                # Performance score based on execution time
                performance_score = 1.0 / (1.0 + evolved_time) if evolved_time > 0 else 0.0
                performance_scores.append(performance_score)
                success_count += 1

            except Exception as e:
                print(f"Trial {{trial}}: Error - {{str(e)}}")
                print(traceback.format_exc())
                continue

        # If all trials failed, return zero scores
        if success_count == 0:
            return {{
                "correctness_score": 0.0,
                "performance_score": 0.0,
                "combined_score": 0.0,
                "speedup_score": 0.0,  # Primary fitness score
                "baseline_comparison": {{
                    "mean_speedup": None,
                    "median_speedup": None,
                    "success_rate": 0.0,
                    "baseline_times": [],
                    "evolved_times": [],
                    "speedups": []
                }},
                "error": "All trials failed",
            }}

        # Calculate metrics
        avg_correctness = float(np.mean(correctness_scores))
        avg_performance = float(np.mean(performance_scores))
        reliability_score = float(success_count / num_trials)

        # Calculate speedup as primary fitness score
        if speedups:
            mean_speedup = float(np.mean(speedups))
            # Use speedup as primary fitness score (higher is better)
            speedup_score = mean_speedup
        else:
            speedup_score = 0.0
            mean_speedup = None

        # Combined score prioritizing correctness (kept for compatibility)
        combined_score = float(
            0.7 * avg_correctness + 0.2 * avg_performance + 0.1 * reliability_score
        )

        # Calculate baseline comparison metrics
        baseline_comparison = {{
            "mean_speedup": mean_speedup,
            "median_speedup": float(np.median(speedups)) if speedups else None,
            "success_rate": float(valid_count / success_count) if success_count > 0 else 0.0,
            "baseline_times": baseline_times,
            "evolved_times": evolved_times,
            "speedups": speedups,
            "num_valid_solutions": valid_count,
            "num_total_trials": success_count
        }}

        return {{
            "correctness_score": avg_correctness,
            "performance_score": avg_performance,
            "reliability_score": reliability_score,
            "combined_score": combined_score,
            "speedup_score": speedup_score,  # Primary fitness score for evolution
            "success_rate": reliability_score,
            "baseline_comparison": baseline_comparison,
        }}

    except Exception as e:
        print(f"Evaluation failed completely: {{str(e)}}")
        print(traceback.format_exc())
        return {{
            "correctness_score": 0.0,
            "performance_score": 0.0,
            "combined_score": 0.0,
            "speedup_score": 0.0,  # Primary fitness score
            "baseline_comparison": {{
                "mean_speedup": None,
                "median_speedup": None,
                "success_rate": 0.0,
                "baseline_times": [],
                "evolved_times": [],
                "speedups": []
            }},
            "error": str(e),
        }}

# Stage-based evaluation for cascade evaluation
def evaluate_stage1(program_path, config=None):
    """First stage evaluation with basic functionality check of the evolved solve method"""
    try:
        # Load configuration
        if config is None:
            config = {{
                "algotune": {{
                    "num_trials": 5,
                    "data_size": 5,
                    "timeout": 30
                }}
            }}
        
        algotune_config = config.get("algotune", {{}})
        data_size = algotune_config.get("data_size", 5)
        timeout_seconds = algotune_config.get("timeout", 30)
        
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if the required function exists
        if not hasattr(program, "run_solver"):
            return {{"runs_successfully": 0.0, "error": "Missing run_solver function"}}

        # Get the original task for reference solutions and problem generation
        task_class = None
        if "{task_name}" in TASK_REGISTRY:
            task_class = TASK_REGISTRY["{task_name}"]
        else:
            print(f"Error: {task_name} task not found in TASK_REGISTRY")
            print(f"Available tasks: {{list(TASK_REGISTRY.keys())}}")

        try:
            # Run a single trial with timeout using proper task-specific problem
            if task_class:
                task_instance = task_class()
                test_problem = task_instance.generate_problem(n=data_size, random_seed=42)
            else:
                # Generic fallback test problem
                test_problem = {{"test_data": [1, 2, 3], "random_seed": 42}}
            
            result = run_with_timeout(program.run_solver, args=(test_problem,), timeout_seconds=timeout_seconds)

            # Basic validity check
            if result is not None:
                return {{
                    "runs_successfully": 1.0,
                    "basic_functionality": 1.0,
                }}
            else:
                return {{
                    "runs_successfully": 0.5,
                    "basic_functionality": 0.0,
                    "error": "Function returned None"
                }}

        except TimeoutError as e:
            return {{"runs_successfully": 0.0, "error": "Timeout"}}
        except Exception as e:
            return {{"runs_successfully": 0.0, "error": str(e)}}

    except Exception as e:
        return {{"runs_successfully": 0.0, "error": str(e)}}

def evaluate_stage2(program_path, config=None):
    """Second stage evaluation with more thorough testing of the evolved solve method"""
    return evaluate(program_path, config)
'''
        
        return evaluator
    
    def _generate_config(self, task_name: str) -> str:
        """Generate the configuration for OpenEvolve with baseline comparison."""
        import re
        
        description = self.get_task_description(task_name)
        
        # Extract category from description
        category = "optimization"  # default
        if "Category:" in description:
            category_line = [line for line in description.split('\n') if line.startswith('Category:')]
            if category_line:
                category = category_line[0].split('Category:')[1].strip()
        
        # Clean up the description for YAML compatibility
        clean_description = description.split('Input:')[0].strip()
        
        # Fix Unicode escape issues in docstrings
        # Replace problematic byte literals with safer representations
        # Use simple string replacement instead of regex for better reliability
        clean_description = clean_description.replace('\\x', '\\\\x')
        clean_description = clean_description.replace('b\\x', 'b\\\\x')
        
        # Generic LaTeX command handling using regex
        
        # Handle LaTeX commands: \command{arg} or \command
        # This regex matches LaTeX commands and replaces them with their command name
        def replace_latex_command(match):
            command = match.group(1)  # The command name without backslash
            return command
        
        # Replace LaTeX commands with their command names
        clean_description = re.sub(r'\\(\w+)(?:\{[^}]*\})?', replace_latex_command, clean_description)
        
        # Handle YAML escape sequences properly
        clean_description = clean_description.replace('\\', '\\\\')
        clean_description = clean_description.replace('"', '\\"')
        clean_description = clean_description.replace('\n', '\\n')
        clean_description = clean_description.replace('\t', '\\t')
        clean_description = clean_description.replace('\r', '\\r')
        
        # Remove any remaining invalid escape sequences and fix common issues
        clean_description = re.sub(r'\\(?!["\\nrt])', '', clean_description)
        
        # Fix common problematic patterns
        clean_description = clean_description.replace('\\....', '...')
        clean_description = clean_description.replace('\\...', '...')
        clean_description = clean_description.replace('\\..', '..')
        
        # Fix mathematical notation that causes YAML issues
        clean_description = clean_description.replace('\\|', '\\\\|')
        clean_description = clean_description.replace('\\{', '\\\\{')
        clean_description = clean_description.replace('\\}', '\\\\}')
        
        # Ensure the description doesn't exceed reasonable length for YAML
        max_length = 1000  # Changed from 1e3 to 1000
        if len(clean_description) > max_length:
            # Try to truncate at a word boundary
            truncated = clean_description[:max_length]
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.8:  # If we can find a space in the last 20%
                clean_description = truncated[:last_space] + "..."
            else:
                # If no good word boundary, truncate and ensure we don't break escape sequences
                clean_description = truncated.rstrip('\\') + "..."
        
        # Insert the new system prompt before the task description
        system_prompt = (
            "SETTING:\n"
            "You're an autonomous programmer tasked with solving a specific problem. You are to use the commands defined below to accomplish this task. Every message you send incurs a cost—you will be informed of your usage and remaining budget by the system.\n"
            "You will be evaluated based on the best-performing piece of code you produce, even if the final code doesn't work or compile (as long as it worked at some point and achieved a score, you will be eligible).\n"
            "Apart from the default Python packages, you have access to the following additional packages:\n"
            " - cryptography\n - cvxpy\n - cython\n - dace\n - dask\n - diffrax\n - ecos\n - faiss-cpu\n - hdbscan\n - highspy\n - jax\n - networkx\n - numba\n - numpy\n - ortools\n - pandas\n - pot\n - psutil\n - pulp\n - pyomo\n - python-sat\n - pythran\n - scikit-learn\n - scipy\n - sympy\n - torch\n"
            "Your primary objective is to optimize the `solve` function to run as as fast as possible, while returning the optimal solution.\n"
            "You will receive better scores the quicker your solution runs, and you will be penalized for exceeding the time limit or returning non-optimal solutions.\n\n"
            "Below you find the description of the task you will have to solve. Read it carefully and understand what the problem is and what your solver should do.\n\n"
        )
        config = f'''# Configuration for {task_name} task with baseline comparison
max_iterations: 100
checkpoint_interval: 10
log_level: "INFO"

# LLM configuration
llm:
  primary_model: "gpt-4o-mini"
  primary_model_weight: 0.8
  secondary_model: "gpt-4o"
  secondary_model_weight: 0.2
  api_base: "https://api.openai.com/v1"
  temperature: 0.7
  top_p: 0.95
  max_tokens: 4096

# Prompt configuration
prompt:
  system_message: "{system_prompt}You are an expert programmer specializing in {category} algorithms. Your task is to improve the {task_name} algorithm implementation with baseline comparison. The problem description is: {clean_description}. Focus on improving the solve method to correctly handle the input format and produce valid solutions efficiently. Your solution will be compared against the reference AlgoTune baseline implementation to measure speedup and correctness."
  num_top_programs: 3
  use_template_stochasticity: true

# Database configuration
database:
  population_size: 50
  archive_size: 20
  num_islands: 3
  elite_selection_ratio: 0.2
  exploitation_ratio: 0.7

# Evaluator configuration
evaluator:
  cascade_evaluation: true
  cascade_thresholds: [0.5, 0.75]
  parallel_evaluations: 4
  use_llm_feedback: false

# AlgoTune task-specific configuration with baseline comparison
algotune:
  num_trials: 5
  data_size: 5
  timeout: 30
  num_runs: 3
  warmup_runs: 1

# Evolution settings
diff_based_evolution: true
allow_full_rewrites: false
'''
        
        return config
    
    def _generate_task_specific_method(self, task_name: str, solve_method: str, class_info: Dict[str, Any]) -> str:
        """Generate a generic fallback method when the actual solve method cannot be extracted."""
        
        # Analyze the solve method to understand the problem structure and return type
        problem_keys = self._extract_problem_keys(solve_method)
        return_type = self._infer_return_type(solve_method, task_name)
        
        return self._generate_generic_method(task_name, problem_keys, return_type)
    
    def _extract_problem_keys(self, solve_method: str) -> List[str]:
        """Extract the expected problem keys from the solve method."""
        keys = []
        if 'problem["X"]' in solve_method:
            keys.append("X")
        if 'problem["y"]' in solve_method:
            keys.append("y")
        if 'problem["k"]' in solve_method:
            keys.append("k")
        if 'problem["C"]' in solve_method:
            keys.append("C")
        if 'problem["matrix"]' in solve_method:
            keys.append("matrix")
        if 'problem["x_data"]' in solve_method:
            keys.append("x_data")
        if 'problem["y_data"]' in solve_method:
            keys.append("y_data")
        if 'problem["model_type"]' in solve_method:
            keys.append("model_type")
        return keys
    
    def _infer_return_type(self, solve_method: str, task_name: str) -> str:
        """Infer the expected return type from the solve method."""
        if '.tolist()' in solve_method:
            return 'list'
        elif 'return {' in solve_method or 'return {' in solve_method:
            return 'dict'
        elif 'return None' in solve_method:
            return 'None'
        else:
            # Generic fallback - analyze based on method content
            return 'unknown'
    
    def _generate_generic_method(self, task_name: str, problem_keys: List[str], return_type: str) -> str:
        """Generate a generic method based on problem structure and return type."""
        
        # Build problem validation
        validation_lines = []
        for key in problem_keys:
            validation_lines.append(f'            if "{key}" not in problem:')
            validation_lines.append(f'                logging.error(f"Problem must contain \'{key}\' key")')
            validation_lines.append(f'                raise ValueError(f"Missing required key: {key}")')
        
        validation_code = '\n'.join(validation_lines) if validation_lines else '            # No specific validation needed'
        
        # Build return statement based on return type
        if return_type == 'list':
            return_code = '            return []  # Placeholder list return'
        elif return_type == 'dict':
            return_code = '            return {}  # Placeholder dict return'
        else:
            return_code = '            return None  # Placeholder return'
        
        return f"""            # Generic implementation for {task_name}
            # Expected problem keys: {problem_keys}
            # Expected return type: {return_type}
            
{validation_code}
            
            # TODO: Implement proper solution for {task_name}
            # This is a placeholder that will be evolved
            logging.warning("Using placeholder implementation - will be evolved")
{return_code}"""
    
    def create_task_files(self, task_name: str, output_dir: Optional[str] = None) -> str:
        """
        Create OpenEvolve files for a specific task.
        
        Args:
            task_name: Name of the AlgoTune task
            output_dir: Output directory (defaults to task_name subdirectory)
            
        Returns:
            Path to the created directory
        """
        if task_name not in self.available_tasks:
            raise ValueError(f"Task '{task_name}' not found. Available tasks: {list(self.available_tasks.keys())}")
        
        if output_dir is None:
            output_dir = self.output_path / task_name
        else:
            output_dir = Path(output_dir)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate files
        initial_program = self._generate_initial_program(task_name)
        evaluator = self._generate_evaluator(task_name)
        config = self._generate_config(task_name)
        
        # Write files
        with open(output_dir / "initial_program.py", "w") as f:
            f.write(initial_program)
        
        with open(output_dir / "evaluator.py", "w") as f:
            f.write(evaluator)
        
        with open(output_dir / "config.yaml", "w") as f:
            f.write(config)
        
        return str(output_dir)
    
    def list_available_tasks(self) -> List[str]:
        """List all available AlgoTune tasks."""
        return list(self.available_tasks.keys())
    
    def get_task_info(self, task_name: str) -> Dict[str, Any]:
        """Get detailed information about a task."""
        if task_name not in self.available_tasks:
            raise ValueError(f"Task '{task_name}' not found")
        
        return {
            'name': task_name,
            'description': self.get_task_description(task_name),
            'path': str(self.available_tasks[task_name]['path']),
            'available': True
        }