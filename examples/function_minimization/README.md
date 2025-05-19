# Function Minimization Example

This example demonstrates how OpenEvolve can discover sophisticated optimization algorithms starting from a simple implementation.

## Problem Description

The task is to minimize a complex non-convex function with multiple local minima:

```python
f(x, y) = sin(x) * cos(y) + sin(x*y) + (x^2 + y^2)/20
```

The global minimum is approximately at (-1.704, 0.678) with a value of -1.519.

## Getting Started

To run this example:

```bash
cd examples/function_minimization
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```

## Algorithm Evolution

### Initial Algorithm (Random Search)

The initial implementation was a simple random search that had no memory between iterations:

```python
def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    A simple random search algorithm that often gets stuck in local minima.
    
    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)
        
    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with a random point
    best_x = np.random.uniform(bounds[0], bounds[1])
    best_y = np.random.uniform(bounds[0], bounds[1])
    best_value = evaluate_function(best_x, best_y)
    
    for _ in range(iterations):
        # Simple random search
        x = np.random.uniform(bounds[0], bounds[1])
        y = np.random.uniform(bounds[0], bounds[1])
        value = evaluate_function(x, y)
        
        if value < best_value:
            best_value = value
            best_x, best_y = x, y
    
    return best_x, best_y, best_value
```

### Evolved Algorithm (Simulated Annealing)

After running OpenEvolve, it discovered a simulated annealing algorithm with a completely different approach:

```python
def simulated_annealing(bounds=(-5, 5), iterations=1000, step_size=0.1, initial_temperature=100, cooling_rate=0.99):
    """
    Simulated Annealing algorithm for function minimization.
    
    Args:
        bounds: Bounds for the search space (min, max)
        iterations: Number of iterations to run
        step_size: Step size for perturbing the solution
        initial_temperature: Initial temperature for the simulated annealing process
        cooling_rate: Cooling rate for the simulated annealing process
        
    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with a random point
    best_x = np.random.uniform(bounds[0], bounds[1])
    best_y = np.random.uniform(bounds[0], bounds[1])
    best_value = evaluate_function(best_x, best_y)

    current_x, current_y = best_x, best_y
    current_value = best_value
    temperature = initial_temperature

    for _ in range(iterations):
        # Perturb the current solution
        new_x = current_x + np.random.uniform(-step_size, step_size)
        new_y = current_y + np.random.uniform(-step_size, step_size)

        # Ensure the new solution is within bounds
        new_x = max(bounds[0], min(new_x, bounds[1]))
        new_y = max(bounds[0], min(new_y, bounds[1]))

        new_value = evaluate_function(new_x, new_y)

        # Calculate the acceptance probability
        if new_value < current_value:
            current_x, current_y = new_x, new_y
            current_value = new_value

            if new_value < best_value:
                best_x, best_y = new_x, new_y
                best_value = new_value
        else:
            probability = np.exp((current_value - new_value) / temperature)
            if np.random.rand() < probability:
                current_x, current_y = new_x, new_y
                current_value = new_value

        # Cool down the temperature
        temperature *= cooling_rate

    return best_x, best_y, best_value
```

## Key Improvements

Through evolutionary iterations, OpenEvolve discovered several key algorithmic concepts:

1. **Local Search**: Instead of random sampling across the entire space, the evolved algorithm makes small perturbations to promising solutions:
   ```python
   new_x = current_x + np.random.uniform(-step_size, step_size)
   new_y = current_y + np.random.uniform(-step_size, step_size)
   ```

2. **Temperature-based Acceptance**: The algorithm can escape local minima by occasionally accepting worse solutions:
   ```python
   probability = np.exp((current_value - new_value) / temperature)
   if np.random.rand() < probability:
       current_x, current_y = new_x, new_y
       current_value = new_value
   ```

3. **Cooling Schedule**: The temperature gradually decreases, transitioning from exploration to exploitation:
   ```python
   temperature *= cooling_rate
   ```

4. **Parameter Introduction**: The system discovered the need for additional parameters to control the algorithm's behavior:
   ```python
   def simulated_annealing(bounds=(-5, 5), iterations=1000, step_size=0.1, initial_temperature=100, cooling_rate=0.99):
   ```

## Results

The evolved algorithm shows substantial improvement in finding better solutions:

| Metric | Value |
|--------|-------|
| Value Score | 0.677 |
| Distance Score | 0.258 |
| Reliability Score | 1.000 |
| Overall Score | 0.917 |
| Combined Score | 0.584 |

The simulated annealing algorithm:
- Achieves higher quality solutions (closer to the global minimum)
- Has perfect reliability (100% success rate in completing runs)
- Maintains a good balance between performance and reliability

## How It Works

This example demonstrates key features of OpenEvolve:

- **Code Evolution**: Only the code inside the evolve blocks is modified
- **Complete Algorithm Redesign**: The system transformed a random search into a completely different algorithm
- **Automatic Discovery**: The system discovered simulated annealing without being explicitly programmed with knowledge of optimization algorithms
- **Function Renaming**: The system even recognized that the algorithm should have a more descriptive name

## Next Steps

Try modifying the config.yaml file to:
- Increase the number of iterations
- Change the LLM model configuration
- Adjust the evaluator settings to prioritize different metrics
- Try a different objective function by modifying `evaluate_function()`
