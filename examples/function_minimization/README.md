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
def search_algorithm(bounds=(-5, 5), iterations=2000, initial_temperature=100, cooling_rate=0.97, step_size_factor=0.2, step_size_increase_threshold=20):
    """
    Simulated Annealing algorithm for function minimization.
    
    Args:
        bounds: Bounds for the search space (min, max)
        iterations: Number of iterations to run
        initial_temperature: Initial temperature for the simulated annealing process
        cooling_rate: Cooling rate for the simulated annealing process
        step_size_factor: Factor to scale the initial step size by the range
        step_size_increase_threshold: Number of iterations without improvement before increasing step size

    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize
    best_x = np.random.uniform(bounds[0], bounds[1])
    best_y = np.random.uniform(bounds[0], bounds[1])
    best_value = evaluate_function(best_x, best_y)

    current_x, current_y = best_x, best_y
    current_value = best_value
    temperature = initial_temperature
    step_size = (bounds[1] - bounds[0]) * step_size_factor  # Initial step size
    min_temperature = 1e-6 # Avoid premature convergence
    no_improvement_count = 0 # Counter for tracking stagnation

    for i in range(iterations):
        # Adaptive step size and temperature control
        if i > iterations * 0.75:  # Reduce step size towards the end
            step_size *= 0.5
        if no_improvement_count > step_size_increase_threshold: # Increase step size if stuck
            step_size *= 1.1
            no_improvement_count = 0 # Reset the counter

        step_size = min(step_size, (bounds[1] - bounds[0]) * 0.5) # Limit step size

        new_x = current_x + np.random.uniform(-step_size, step_size)
        new_y = current_y + np.random.uniform(-step_size, step_size)

        # Keep the new points within the bounds
        new_x = max(bounds[0], min(new_x, bounds[1]))
        new_y = max(bounds[0], min(new_y, bounds[1]))

        new_value = evaluate_function(new_x, new_y)

        if new_value < current_value:
            # Accept the move if it's better
            current_x, current_y = new_x, new_y
            current_value = new_value
            no_improvement_count = 0  # Reset counter

            if new_value < best_value:
                # Update the best found solution
                best_x, best_y = new_x, new_y
                best_value = new_value
        else:
            # Accept with a certain probability (Simulated Annealing)
            probability = np.exp((current_value - new_value) / temperature)
            if np.random.rand() < probability:
                current_x, current_y = new_x, new_y
                current_value = new_value
                no_improvement_count = 0  # Reset counter
            else:
                no_improvement_count += 1 # Increment counter if not improving

        temperature = max(temperature * cooling_rate, min_temperature) #Cool down

    return best_x, best_y, best_value
```

## Key Improvements

Through evolutionary iterations, OpenEvolve discovered several key algorithmic concepts:

1. **Exploration via Temperature**: Simulated annealing uses a `temperature` parameter to allow uphill moves early in the search, helping escape local minima that would trap simpler methods.
    ```python
    probability = np.exp((current_value - new_value) / temperature)
    ```

2. **Adaptive Step Size**: The step size is adjusted dynamically—shrinking as the search converges and expanding if progress stalls—leading to better coverage and faster convergence.
    ```python
    if i > iterations * 0.75:  # Reduce step size towards the end
        step_size *= 0.5
    if no_improvement_count > step_size_increase_threshold: # Increase step size if stuck
        step_size *= 1.1
        no_improvement_count = 0 # Reset the counter
    ```

3. **Bounded Moves**: The algorithm ensures all candidate solutions remain within the feasible domain, avoiding wasted evaluations.
    ```python
    # Keep the new points within the bounds
    new_x = max(bounds[0], min(new_x, bounds[1]))
    new_y = max(bounds[0], min(new_y, bounds[1]))
    ```

4. **Stagnation Handling**: By counting iterations without improvement, the algorithm responds by boosting exploration when progress stalls.
    ```python
    if no_improvement_count > step_size_increase_threshold: # Increase step size if stuck
        step_size *= 1.1
        no_improvement_count = 0 # Reset the counter
    ```

## Results

The evolved algorithm shows substantial improvement in finding better solutions:

| Metric | Value |
|--------|-------|
| Value Score | 0.990 |
| Distance Score | 0.921 |
| Standard Deviation Score | 0.900 |
| Speed Score | 0.466 |
| Reliability Score | 1.000 |
| Overall Score | 0.984 |
| Combined Score | 0.922 |

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
