"""
Evaluator for the function minimization example
"""

import re
import subprocess
import time
import traceback


def run_with_timeout(program_path, timeout_seconds=60):
    """
    Run a function with a timeout using subprocess.

    Args:
        func: Function to run
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        timeout_seconds: Timeout in seconds

    Returns:
        Result of the function or raises TimeoutError
    """
    cmd = ["python", "submit.py", program_path, "-p", "alphabet", "-l", "Python 3", "-f"]

    try:
        # Run the command and grab its output using subprocess.Popen
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
        exit_code = proc.returncode
        if exit_code != 0:
            print(stderr)  # Print the error output if the command failed
            raise RuntimeError(f"Process exited with code {exit_code}")
    except subprocess.TimeoutExpired:
        # Kill the process if it times out
        proc.kill()
        raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    pattern = (
        r"Score:\s*(\d+)\s*"
        r"Test cases done:\s*(\d+)\s*"
        r"Test cases correct:\s*(\d+)\s*"
        r"Test cases total:\s*(\d+)"
    )
    match = re.search(pattern, stdout)
    if not match:
        raise ValueError("Expected summary lines not found")

    score, done, correct, total = map(int, match.groups())
    return score, done, correct, total


def evaluate(program_path):
    """
    Evaluate the program by submitting it to OJ and fetching metrics based on how well it performs.

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    try:
        # For constructor-based approaches, a single evaluation is sufficient
        # since the result is deterministic
        start_time = time.time()

        # Use subprocess to run with timeout
        score, done, correct, total = run_with_timeout(
            program_path, timeout_seconds=60  # Single timeout
        )

        end_time = time.time()
        eval_time = end_time - start_time

        # Combined score - higher is better
        combined_score = correct / total if total > 0 else 0.0

        print(
            f"Evaluation: Score={score}, Done={done}, Correct={correct}, Total={total}, Combined={combined_score:.2f}"
        )

        return {
            "score": score,
            "done": done,
            "correct": correct,
            "total": total,
            "eval_time": eval_time,
            "combined_score": float(combined_score),
        }

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        traceback.print_exc()
        return {
            "score": 0,
            "done": 0,
            "correct": 0,
            "total": 0,
            "eval_time": 0.0,
            "combined_score": 0.0,
        }
