# Rust Adaptive Sorting Evolution

This example demonstrates how to use OpenEvolve with the Rust programming language. The example focuses on evolving adaptive sorting algorithms that optimize their behavior based on input data characteristics, showcasing OpenEvolve's ability to work with compiled systems programming languages.

## Files

- `initial_program.rs`: Starting Rust implementation with basic quicksort
- `evaluator.py`: Python evaluator that compiles and benchmarks Rust code
- `config.yaml`: Configuration optimized for performance-critical algorithm evolution
- `requirements.txt`: System dependencies and Python requirements

## Prerequisites

### System Dependencies
1. **Rust Toolchain**: Install from [rustup.rs](https://rustup.rs/)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   ```

2. **Cargo**: Comes with Rust installation

### Python Dependencies
```bash
pip install -r requirements.txt
```

## Usage

Run the evolution process:

```bash
cd examples/rust_adaptive_sort
python ../../openevolve-run.py initial_program.rs evaluator.py --config config.yaml --iterations 150
```

This example shows how OpenEvolve can evolve algorithms in Rust by starting with a basic quicksort implementation and improving it to adaptively handle different data patterns and optimize performance across various sorting scenarios.