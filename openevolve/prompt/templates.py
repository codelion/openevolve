"""
Prompt templates for OpenEvolve
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# Base system message template for evolution
BASE_SYSTEM_TEMPLATE = """You are an expert software developer tasked with iteratively improving a codebase.
Your job is to analyze the current program and suggest improvements based on feedback from previous attempts.
Focus on making targeted changes that will increase the program's performance metrics.
"""

# Matrix multiplication system template
MATMUL_SYSTEM_TEMPLATE = """You are an expert algorithm engineer specialized in numerical computing and matrix operations with a deep expertise in matrix multiplication optimizations.

Your task is to optimize matrix multiplication algorithms for better performance while maintaining correctness. You're familiar with advanced techniques including:

1. Strassen's algorithm, which reduces 7 multiplications instead of 8 for 2x2 matrices
2. Winograd's variant, which minimizes additions in Strassen's algorithm
3. The Coppersmith-Winograd algorithm and its theoretical improvements
4. Memory access pattern optimizations (loop reordering, cache-oblivious algorithms)
5. Low-level optimizations (loop unrolling, SIMD-friendly code, elimination of unnecessary operations)
6. Special case optimizations for specific matrix dimensions
7. Advanced mathematical decompositions like tensor methods

Focus particularly on optimizing small matrix sizes (2x2 to 5x5) where algorithmic innovations can make a significant difference versus hardware-level optimizations. Apply insights from linear algebra to reduce the total number of operations required.

The goal is to achieve the maximum possible speedup while maintaining 100% correctness of the output compared to the standard implementation.
"""

# User message template for diff-based evolution
DIFF_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Suggest improvements to the program that will lead to better performance on the specified metrics.

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Example of valid diff format:
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.
Be thoughtful about your changes and explain your reasoning thoroughly.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
"""

# Matrix multiplication specific template
MATMUL_DIFF_USER_TEMPLATE = """# Matrix Multiplication Optimization Task
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Optimize the matrix multiplication algorithm for better performance while maintaining correctness.
Your goal is to achieve the maximum possible speedup for matrix sizes from 2x2 to 5x5.

The evaluation metrics show how much your implementation is faster than the naive algorithm. Higher values are better. The optimization techniques you should consider include:

## Algorithm-level optimizations (highest impact):
1. Implement Strassen's algorithm for 2x2, 4x4 matrices (reduces operations from O(n³) to O(n²·⁸¹))
2. Create specialized functions for specific matrix sizes (2x2, 3x3, 4x4, 5x5)
3. Recursive decomposition with custom base cases
4. Winograd's variant that minimizes the number of additions
5. Tensor-based decompositions for further reducing scalar multiplications

## Implementation-level optimizations:
1. Loop reordering for better cache locality (k-i-j instead of i-j-k)
2. Loop unrolling to reduce loop overhead and enable compiler optimizations
3. Memory access pattern improvements (array layout, temporary storage)
4. Complete elimination of unnecessary operations and checks
5. Smart bounds checking and early termination for special cases

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Examples of good changes include:

1. Implementing Strassen for 2x2 matrices:
<<<<<<< SEARCH
def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    m, n = A.shape
    n2, p = B.shape
    
    if n != n2:
        raise ValueError(f"Incompatible matrix shapes: {{A.shape}} and {{B.shape}}")
    
    # Initialize result matrix with zeros
    C = np.zeros((m, p), dtype=A.dtype)
    
    # Naive triple-loop implementation
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C
=======
def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    m, n = A.shape
    n2, p = B.shape
    
    if n != n2:
        raise ValueError(f"Incompatible matrix shapes: {{A.shape}} and {{B.shape}}")
    
    # Special case for 2x2 matrices using Strassen's algorithm
    if m == 2 and n == 2 and p == 2:
        return strassen_2x2(A, B)
    
    # Initialize result matrix with zeros
    C = np.zeros((m, p), dtype=A.dtype)
    
    # Optimized loop ordering for better cache locality
    for i in range(m):
        for k in range(n):
            A_ik = A[i, k]
            for j in range(p):
                C[i, j] += A_ik * B[k, j]
    
    return C

def strassen_2x2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Strassen's algorithm for 2x2 matrices
    # This reduces multiplications from 8 to 7
    
    # Extract elements
    a11, a12 = A[0, 0], A[0, 1]
    a21, a22 = A[1, 0], A[1, 1]
    b11, b12 = B[0, 0], B[0, 1]
    b21, b22 = B[1, 0], B[1, 1]
    
    # Compute the 7 products needed in Strassen's algorithm
    m1 = (a11 + a22) * (b11 + b22)
    m2 = (a21 + a22) * b11
    m3 = a11 * (b12 - b22)
    m4 = a22 * (b21 - b11)
    m5 = (a11 + a12) * b22
    m6 = (a21 - a11) * (b11 + b12)
    m7 = (a12 - a22) * (b21 + b22)
    
    # Compute the result matrix elements
    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6
    
    # Construct the result matrix
    C = np.zeros((2, 2), dtype=A.dtype)
    C[0, 0], C[0, 1] = c11, c12
    C[1, 0], C[1, 1] = c21, c22
    
    return C
>>>>>>> REPLACE

Explain your reasoning and clearly state which specific optimizations you're implementing. Be creative but thorough in your approach to achieve the maximum possible speedup.
"""

# User message template for full rewrite
FULL_REWRITE_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

# Program Evolution History
{evolution_history}

# Current Program
```{language}
{current_program}
```

# Task
Rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs
as the original program, but with improved internal implementation.

```{language}
# Your rewritten program here
```
"""

# Template for formatting evolution history
EVOLUTION_HISTORY_TEMPLATE = """## Previous Attempts

{previous_attempts}

## Top Performing Programs

{top_programs}
"""

# Template for formatting a previous attempt
PREVIOUS_ATTEMPT_TEMPLATE = """### Attempt {attempt_number}
- Changes: {changes}
- Performance: {performance}
- Outcome: {outcome}
"""

# Template for formatting a top program
TOP_PROGRAM_TEMPLATE = """### Program {program_number} (Score: {score})
```{language}
{program_snippet}
```
Key features: {key_features}
"""

# Default templates dictionary
DEFAULT_TEMPLATES = {
    "system_message": BASE_SYSTEM_TEMPLATE,
    "matmul_system": MATMUL_SYSTEM_TEMPLATE,
    "diff_user": DIFF_USER_TEMPLATE,
    "matmul_diff_user": MATMUL_DIFF_USER_TEMPLATE,
    "full_rewrite_user": FULL_REWRITE_USER_TEMPLATE,
    "evolution_history": EVOLUTION_HISTORY_TEMPLATE,
    "previous_attempt": PREVIOUS_ATTEMPT_TEMPLATE,
    "top_program": TOP_PROGRAM_TEMPLATE,
}


class TemplateManager:
    """Manages templates for prompt generation"""
    
    def __init__(self, template_dir: Optional[str] = None):
        self.templates = DEFAULT_TEMPLATES.copy()
        
        # Load templates from directory if provided
        if template_dir and os.path.isdir(template_dir):
            self._load_templates_from_dir(template_dir)
    
    def _load_templates_from_dir(self, template_dir: str) -> None:
        """Load templates from a directory"""
        for file_path in Path(template_dir).glob("*.txt"):
            template_name = file_path.stem
            with open(file_path, "r") as f:
                self.templates[template_name] = f.read()
    
    def get_template(self, template_name: str) -> str:
        """Get a template by name"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self.templates[template_name]
    
    def add_template(self, template_name: str, template: str) -> None:
        """Add or update a template"""
        self.templates[template_name] = template
