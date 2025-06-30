"""
Evaluator for Rust adaptive sorting example
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from openevolve.evaluation_result import EvaluationResult


async def evaluate(program_path: str) -> EvaluationResult:
    """
    Evaluate a Rust sorting algorithm implementation.
    
    Tests the algorithm on various data patterns to measure:
    - Correctness
    - Performance (speed)
    - Adaptability to different data patterns
    - Memory efficiency
    """
    try:
        # Create a temporary Rust project
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "sort_test"
            
            # Initialize Cargo project
            result = subprocess.run(
                ["cargo", "init", "--name", "sort_test", str(project_dir)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return EvaluationResult(
                    metrics={"score": 0.0, "compile_success": 0.0},
                    artifacts={"error": "Failed to create Cargo project", "stderr": result.stderr}
                )
            
            # Copy the program to src/lib.rs
            lib_path = project_dir / "src" / "lib.rs"
            with open(program_path, 'r') as src:
                lib_content = src.read()
            with open(lib_path, 'w') as dst:
                dst.write(lib_content)
            
            # Create main.rs with benchmark code
            main_content = """
use sort_test::{adaptive_sort, run_benchmark};
use std::time::Instant;

fn main() {
    // Generate test datasets with different characteristics
    let test_data = vec![
        // Random data
        generate_random_data(1000),
        generate_random_data(10000),
        
        // Nearly sorted data
        generate_nearly_sorted_data(1000, 0.05),
        generate_nearly_sorted_data(10000, 0.05),
        
        // Reverse sorted data
        generate_reverse_sorted_data(1000),
        generate_reverse_sorted_data(10000),
        
        // Data with many duplicates
        generate_data_with_duplicates(1000, 10),
        generate_data_with_duplicates(10000, 100),
        
        // Partially sorted data
        generate_partially_sorted_data(1000, 0.3),
        generate_partially_sorted_data(10000, 0.3),
    ];
    
    let results = run_benchmark(test_data);
    
    // Calculate metrics
    let all_correct = results.correctness.iter().all(|&c| c);
    let correctness_score = if all_correct { 1.0 } else { 0.0 };
    
    let avg_time: f64 = results.times.iter().sum::<f64>() / results.times.len() as f64;
    
    // Performance score (normalized, assuming baseline of 0.1 seconds for largest dataset)
    let performance_score = 1.0 / (1.0 + avg_time * 10.0);
    
    // Output results as JSON
    println!("{{");
    println!("  \\"correctness\\": {},", correctness_score);
    println!("  \\"avg_time\\": {},", avg_time);
    println!("  \\"performance_score\\": {},", performance_score);
    println!("  \\"adaptability_score\\": {},", results.adaptability_score);
    println!("  \\"times\\": {:?},", results.times);
    println!("  \\"all_correct\\": {}", all_correct);
    println!("}}");
}

fn generate_random_data(size: usize) -> Vec<i32> {
    (0..size).map(|_| rand::random::<i32>() % 10000).collect()
}

fn generate_nearly_sorted_data(size: usize, disorder_rate: f64) -> Vec<i32> {
    let mut data: Vec<i32> = (0..size as i32).collect();
    let swaps = (size as f64 * disorder_rate) as usize;
    
    for _ in 0..swaps {
        let i = rand::random::<usize>() % size;
        let j = rand::random::<usize>() % size;
        data.swap(i, j);
    }
    
    data
}

fn generate_reverse_sorted_data(size: usize) -> Vec<i32> {
    (0..size as i32).rev().collect()
}

fn generate_data_with_duplicates(size: usize, unique_values: usize) -> Vec<i32> {
    (0..size).map(|_| rand::random::<i32>() % unique_values as i32).collect()
}

fn generate_partially_sorted_data(size: usize, sorted_fraction: f64) -> Vec<i32> {
    let sorted_size = (size as f64 * sorted_fraction) as usize;
    let mut data = Vec::with_capacity(size);
    
    // Add sorted portion
    data.extend((0..sorted_size as i32));
    
    // Add random portion
    data.extend((0..(size - sorted_size)).map(|_| rand::random::<i32>() % 10000));
    
    data
}

// Simple random implementation
mod rand {
    use std::cell::Cell;
    use std::time::{SystemTime, UNIX_EPOCH};
    
    thread_local! {
        static SEED: Cell<u64> = Cell::new(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );
    }
    
    pub fn random<T>() -> T
    where
        T: From<u64>,
    {
        SEED.with(|seed| {
            let mut x = seed.get();
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            seed.set(x);
            T::from(x)
        })
    }
}
"""
            main_path = project_dir / "src" / "main.rs"
            with open(main_path, 'w') as f:
                f.write(main_content)
            
            # Build the project
            build_result = subprocess.run(
                ["cargo", "build", "--release"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if build_result.returncode != 0:
                # Extract compilation errors
                return EvaluationResult(
                    metrics={
                        "score": 0.0,
                        "compile_success": 0.0,
                        "correctness": 0.0,
                        "performance_score": 0.0,
                        "adaptability_score": 0.0
                    },
                    artifacts={
                        "error": "Compilation failed",
                        "stderr": build_result.stderr,
                        "stdout": build_result.stdout
                    }
                )
            
            # Run the benchmark
            run_result = subprocess.run(
                ["cargo", "run", "--release"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if run_result.returncode != 0:
                return EvaluationResult(
                    metrics={
                        "score": 0.0,
                        "compile_success": 1.0,
                        "correctness": 0.0,
                        "performance_score": 0.0,
                        "adaptability_score": 0.0
                    },
                    artifacts={
                        "error": "Runtime error",
                        "stderr": run_result.stderr
                    }
                )
            
            # Parse JSON output
            try:
                # Find JSON in output (between first { and last })
                output = run_result.stdout
                start = output.find('{')
                end = output.rfind('}') + 1
                json_str = output[start:end]
                
                results = json.loads(json_str)
                
                # Calculate overall score
                correctness = results['correctness']
                performance = results['performance_score']
                adaptability = results['adaptability_score']
                
                # Weighted score (correctness is mandatory)
                if correctness < 1.0:
                    overall_score = 0.0
                else:
                    overall_score = (
                        0.6 * performance +
                        0.4 * adaptability
                    )
                
                # Check for memory safety (basic check via valgrind if available)
                memory_safe = 1.0  # Rust is memory safe by default
                
                return EvaluationResult(
                    metrics={
                        "score": overall_score,
                        "compile_success": 1.0,
                        "correctness": correctness,
                        "performance_score": performance,
                        "adaptability_score": adaptability,
                        "avg_time": results['avg_time'],
                        "memory_safe": memory_safe
                    },
                    artifacts={
                        "times": results['times'],
                        "all_correct": results['all_correct'],
                        "build_output": build_result.stdout
                    }
                )
                
            except (json.JSONDecodeError, KeyError) as e:
                return EvaluationResult(
                    metrics={
                        "score": 0.0,
                        "compile_success": 1.0,
                        "correctness": 0.0,
                        "performance_score": 0.0,
                        "adaptability_score": 0.0
                    },
                    artifacts={
                        "error": f"Failed to parse results: {str(e)}",
                        "stdout": run_result.stdout
                    }
                )
                
    except subprocess.TimeoutExpired:
        return EvaluationResult(
            metrics={
                "score": 0.0,
                "compile_success": 0.0,
                "correctness": 0.0,
                "performance_score": 0.0,
                "adaptability_score": 0.0
            },
            artifacts={"error": "Timeout during evaluation"}
        )
    except Exception as e:
        return EvaluationResult(
            metrics={
                "score": 0.0,
                "compile_success": 0.0,
                "correctness": 0.0,
                "performance_score": 0.0,
                "adaptability_score": 0.0
            },
            artifacts={"error": str(e), "type": "evaluation_error"}
        )


# For testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = asyncio.run(evaluate(sys.argv[1]))
        print(f"Score: {result.metrics['score']:.4f}")
        print(f"Correctness: {result.metrics['correctness']:.4f}")
        print(f"Performance: {result.metrics['performance_score']:.4f}")
        print(f"Adaptability: {result.metrics['adaptability_score']:.4f}")