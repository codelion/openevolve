# Matrix Multiplication via Tensor Decomposition

This example demonstrates how to use OpenEvolve to discover efficient matrix multiplication algorithms through tensor decomposition, as described in the AlphaEvolve paper.

This implementation uses PyTorch for tensor operations and optimization, making it widely compatible across different platforms.

## Background

Matrix multiplication is a fundamental operation in computational mathematics and computer science. The naive algorithm for multiplying two matrices requires O(n³) scalar multiplications, but more efficient algorithms exist. In 1969, Volker Strassen discovered an algorithm that requires only 7 scalar multiplications to multiply 2×2 matrices, which leads to an O(n^log₂(7)) ≈ O(n^2.81) algorithm when applied recursively.

The key insight in this example is that matrix multiplication can be represented as a 3D tensor, and finding a low-rank decomposition of this tensor directly translates to discovering an efficient matrix multiplication algorithm. The rank of the decomposition equals the number of scalar multiplications needed.

## Implementation

This example evolves a tensor decomposition algorithm that aims to find minimum-rank decompositions for matrix multiplication tensors. The decomposition process uses gradient-based optimization with various techniques like:

- Regularization to encourage sparse, interpretable solutions
- Specialized initialization strategies for better convergence
- Adaptive learning rates and optimization techniques
- Methods to promote integer or half-integer coefficients
- Techniques to improve numerical stability and convergence
- Cyclical annealing schedules and noise injection for exploration

## Installation

The matrix multiplication example requires additional dependencies beyond the base OpenEvolve requirements. These include PyTorch for efficient tensor operations and gradient-based optimization.

### Option 1: Using the setup script

```bash
# Make the setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

### Option 2: Manual installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# Install the main OpenEvolve package in development mode
pip install -e ..

# Install example-specific dependencies
pip install -r requirements.txt
```

## Running the Example

To run the example:

```bash
python optimize.py --iterations 100 --output output
```

Parameters:
- `--iterations`: Number of evolutionary iterations (default: 100)
- `--output`: Directory to store results (default: "output")
- `--config`: Path to a custom configuration file (optional)

Note: The first run might take longer as PyTorch compiles the computation graph.

## Evaluation

The implementation is evaluated on several key aspects:

1. **Correctness**: Whether the decomposition accurately reconstructs the original tensor
2. **Rank Quality**: How low of a rank the algorithm can achieve for different matrix sizes
3. **Time Efficiency**: How quickly the algorithm converges to a valid decomposition

The examples include various matrix sizes, with special emphasis on cases where improvements over known bounds are possible.

## Expected Results

After evolution, you should expect to see improvements in:

- The ability to find lower-rank decompositions for various matrix sizes
- More reliable convergence even with lower ranks
- Faster convergence to valid decompositions
- Decompositions with simpler (integer or half-integer) coefficients

The ultimate goal is to match or beat the state-of-the-art results mentioned in the AlphaEvolve paper, such as:
- 4×4 complex matrices: Finding a rank-48 decomposition (better than Strassen's recursive algorithm which requires 49)
- Various rectangular matrices: Improving on previously known bounds

## References

- Strassen, V. (1969). "Gaussian elimination is not optimal". Numerische Mathematik, 13(4), 354-356.
- AlphaEvolve paper: "AlphaEvolve: A coding agent for scientific and algorithmic discovery" (2025)
