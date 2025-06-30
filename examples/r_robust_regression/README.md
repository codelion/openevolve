# R Robust Regression Evolution

This example demonstrates how to use OpenEvolve with the R programming language. The example focuses on evolving robust regression algorithms that can handle outliers in data, showcasing OpenEvolve's ability to work with statistical computing languages beyond Python.

## Files

- `initial_program.r`: Starting R implementation with basic least squares regression
- `evaluator.py`: Python evaluator that runs R code and measures performance
- `config.yaml`: Configuration optimized for statistical algorithm evolution
- `requirements.txt`: Dependencies for both R and Python components

## Prerequisites

### R Dependencies
Install R (version 3.6 or higher) and the required packages:

```r
install.packages(c("jsonlite"))
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

## Usage

Run the evolution process:

```bash
cd examples/r_robust_regression
python ../../openevolve-run.py initial_program.r evaluator.py --config config.yaml --iterations 100
```

This example shows how OpenEvolve can evolve algorithms in R by starting with a basic least squares implementation and improving it to handle outliers through various robust regression techniques.