# Evolving Symbolic Regression with OpenEvolve on LLM-SRBench 🧬🔍

This example demonstrates how **OpenEvolve** can be utilized to perform **symbolic regression** tasks using the **[LLM-SRBench benchmark](https://arxiv.org/pdf/2504.10415)**. It showcases OpenEvolve's capability to evolve Python code, transforming simple mathematical expressions into more complex and accurate models that fit given datasets.

------

## 🎯 Problem Description: Symbolic Regression on LLM-SRBench

**Symbolic Regression** is the task of discovering a mathematical expression that best fits a given dataset. Unlike traditional regression techniques that optimize parameters for a predefined model structure, symbolic regression aims to find both the **structure of the model** and its **parameters**.

This example leverages **LLM-SRBench**, a benchmark specifically designed for Large Language Model-based Symbolic Regression. The core objective is to use OpenEvolve to evolve an initial, often simple, model (e.g., a linear model) into a more sophisticated symbolic expression. This evolved expression should accurately capture the underlying relationships within various scientific datasets provided by the benchmark.

------

## 🚀 Getting Started

Follow these steps to set up and run the symbolic regression benchmark example:

### 1. Configure API Keys

The API key is read from the environment `OPENAI_API_KEY` by default. The primary and secondary model we used in testing LLM-SRBench is `gpt-4o` and `o3`. You can check `create_config()` in `data_api.py`.


### 2. Load Benchmark Tasks & Generate Initial Programs

The `data_api.py` script is crucial for setting up the environment. It prepares tasks from the LLM-SRBench dataset (defined by classes in `./bench`, and will be located at `./problems`).

For each benchmark task, this script will automatically generate:

- `initial_program.py`: A starting Python program, typically a simple linear model.
- `evaluator.py`: A tailored evaluation script for the task.
- `config.yaml`: An OpenEvolve configuration file specific to the task.

Run the script from your terminal:

```bash
python data_api.py
```

This will create subdirectories for each benchmark task, populated with the necessary files.

### 3. Run OpenEvolve

Use the provided shell script `scripts.sh` to execute OpenEvolve across the generated benchmark tasks. This script iterates through the task-specific configurations and applies the evolutionary process.

```bash
bash scripts.sh
```

### 4. Evaluate Results

After OpenEvolve has completed its runs, you can evaluate the performance on different subsets of tasks (e.g., bio, chemical, physics, material). The `eval.py` script collates the results and provides a summary.

```bash
python eval.py <subset_path>
```

For example, to evaluate results for the 'physics' subset located in `./problems/phys_osc/`, you would run:

```bash
python eval.py ./problems/phys_osc
```

This script will also save a `JSON` file containing detailed results for your analysis.

------

## 🌱 Algorithm Evolution: From Linear Model to Complex Expression

OpenEvolve works by iteratively modifying an initial Python program to find a better-fitting mathematical expression.

### Initial Algorithm (Example: Linear Model)

The `data_api.py` script typically generates a basic linear model as the starting point. For a given task, this `initial_program.py` might look like this:

```python
"""
Initial program: A naive linear model for symbolic regression.
This model predicts the output as a linear combination of input variables
or a constant if no input variables are present.
The function is designed for vectorized input (X matrix).

Target output variable: dv_dt (Acceleration in Nonl-linear Harmonic Oscillator)
Input variables (columns of x): x (Position at time t), t (Time), v (Velocity at time t)
"""
import numpy as np

# Input variable mapping for x (columns of the input matrix):
#   x[:, 0]: x (Position at time t)
#   x[:, 1]: t (Time)
#   x[:, 2]: v (Velocity at time t)

# Parameters will be optimized by BFGS outside this function.
# Number of parameters expected by this model: 10.
# Example initialization: params = np.random.rand(10)

# EVOLVE-BLOCK-START

def func(x, params):
    """
    Calculates the model output using a linear combination of input variables
    or a constant value if no input variables. Operates on a matrix of samples.

    Args:
        x (np.ndarray): A 2D numpy array of input variable values, shape (n_samples, n_features).
                        n_features is 3.
                        If n_features is 0, x should be shape (n_samples, 0).
                        The order of columns in x must correspond to:
                        (x, t, v).
        params (np.ndarray): A 1D numpy array of parameters.
                             Expected length: 10.

    Returns:
        np.ndarray: A 1D numpy array of predicted output values, shape (n_samples,).
    """

    result = x[:, 0] * params[0] + x[:, 1] * params[1] + x[:, 2] * params[2]
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
# It ensures that OpenEvolve can consistently call the evolving function.
def run_search():
    return func

# Note: The actual structure of initial_program.py is determined by data_api.py.
```

### Evolved Algorithm (Discovered Symbolic Expression)

**OpenEvolve** iteratively modifies Python code segments, delineated by `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` markers within an `initial_program.py` file. The primary objective is to evolve a simple initial model into a more complex and accurate symbolic expression that minimizes the Mean Squared Error (MSE) against the training data.

Below is a symbolic expression discovered by OpenEvolve for the physics task `PO10`:

```python
import numpy as np

def func(x, params):
    """
    Calculates the model output using a linear combination of input variables
    or a constant value if no input variables. Operates on a matrix of samples.

    Args:
        x (np.ndarray): A 2D numpy array of input variable values, shape (n_samples, n_features).
                        n_features is 2.
                        If n_features is 0, x should be shape (n_samples, 0).
                        The order of columns in x must correspond to:
                        (x, t).
        params (np.ndarray): A 1D numpy array of parameters.
                             Expected length: 10.

    Returns:
        np.ndarray: A 1D numpy array of predicted output values, shape (n_samples,).
    """
    # --------------------------------------------------------------------------
    # Allow for flexible parameter count, only padding essential parts.
    if len(params) < 10:
        required_params = params.shape[0]
        params = np.pad(params, (0, 10 - required_params))

    # Readable aliases for the two input features
    pos = x[:, 0]       # position   x(t)
    t_val = x[:, 1]     # time       t

    # ----------   Internal restoring forces (Duffing-like)   ------------------
    # −k x −β x³ −γ x⁵    (only odd powers, respecting the usual symmetry)
    # Reduced polynomial order (up to cubic) to avoid over-fitting while
    # still capturing the essential softening/stiffening behaviour.
    restoring = -(params[0] * pos + params[1] * pos**3)

    # ----------   Externally forced, periodically driven term   --------------
    #  A e^{-λ t} sin(ω t)   +   B cos(Ω t)   (General form considered)
    # Let the optimiser decide whether the envelope should grow
    # or decay by keeping the sign of params[4].  The exponent is
    # clipped to avoid numerical overflow.
    # Simple periodic forcing without exponential envelope.  This is
    # sufficient for many driven oscillator benchmarks and reduces the
    # risk of numerical overflow in exp().
    trig1 = params[3] * t_val
    trig2 = params[5] * t_val
    forcing = params[2] * np.cos(trig1) + params[4] * np.sin(trig2)

    # ----------   Weak position–time coupling & constant bias   ---------------
    interaction = params[8] * pos * t_val
    bias = params[9]

    return restoring + forcing + interaction + bias
```

The ground truth for this PO10 task is represented by the equation: 

$F_0sin(t)−ω_0^2(γt+1)x(t)−ω_0^2x(t)^3−ω_0^2x(t).$

This can be expanded and simplified to:

$F_0sin(t)−ω_0^2γtx(t)−2ω_0^2x(t)−ω_0^2x(t)^3.$

Notably, the core functional forms present in this ground truth equation are captured by the evolved symbolic expression:

- The $sin(t)$ component can be represented by `params[4] * np.sin(params[5] * t_val)`.
- The linear $x(t)$ term corresponds to `params[0] * pos`.
- The cubic $x(t)^3$ term is `params[1] * pos**3`.
- The interaction term $t⋅x(t)$ is captured by `params[8] * pos * t_val`.

The evolved code also includes terms like `params[2] * np.cos(params[3] * t_val)` (a cosine forcing term) and `params[9]` (a constant bias). These might evolve to have negligible parameter values if not supported by the data, or they could capture secondary effects or noise. The inclusion of the primary terms demonstrates OpenEvolve's strength in identifying the correct underlying structure of the equation.

*Note: Symbolic regression, despite such promising results, remains a very challenging task. This difficulty largely stems from the inherent complexities of inferring precise mathematical models from finite and potentially noisy training data, which provides only a partial observation of the true underlying system.*

------

## ⚙️ Key Configuration & Approach

- LLM Models:
  - **Primary Model:** `gpt-4o` (or your configured `primary_model`) is typically used for sophisticated code generation and modification.
  - **Secondary Model:** `o3` (or your configured `secondary_model`) can be used for refinements, simpler modifications, or other auxiliary tasks within the evolutionary process.
- Evaluation Strategy:
  - Currently, this example employs a direct evaluation strategy (not **cascade evaluation**).
- Objective Function:
  - The primary objective is to **minimize the Mean Squared Error (MSE)** between the model's predictions and the true values on the training data.

------

## 📊 Results

The `eval.py` script will help you collect and analyze performance metrics. The LLM-SRBench paper provides a comprehensive comparison of various baselines. For results generated by this specific OpenEvolve example, you should run the evaluation script as described in the "Getting Started" section.

For benchmark-wide comparisons and results from other methods, please refer to the official LLM-SRBench paper.

*Note: Below we extract the approximate results of baselines in Fig.5 from LLMSR-Bench paper.*

**Median NMSE (Test Set)**

| **Domain**       | **Direct**  | **LLMSR**       | **LaSR**    | **SGA**     | **OpenEvolve** |
| ---------------- | ----------- | --------------- | ----------- | ----------- | -------------- |
| Chemistry        | ~6.0 × 10⁻¹ | **~1.5 × 10⁻⁶** | ~1.0 × 10⁻⁴ | ~1.0 × 10⁻² | 2.34 × 10⁻⁶    |
| Biology          | ~2.0 × 10⁻² | ~1.0 × 10⁻⁵     | ~1.0 × 10⁻⁴ | ~2.0 × 10⁻⁴ | –              |
| Physics          | ~3.0 × 10⁻¹ | **~2.0 × 10⁻⁷** | ~1.0 × 10⁻³ | ~4.0 × 10⁻³ | 1.85 × 10⁻⁵    |
| Material Science | ~3.0 × 10⁻¹ | ~1.0 × 10⁻⁴     | ~7.0 × 10⁻⁴ | ~3.0 × 10⁻² | –              |

**Median NMSE (OOD Test Set)**

| **Domain**       | **Direct** | **LLMSR**   | **LaSR**    | **SGA**    | **OpenEvolve**  |
| ---------------- | ---------- | ----------- | ----------- | ---------- | --------------- |
| Chemistry        | ~3.0 × 10² | ~5.0 × 10⁻² | ~1.0 × 10⁰  | ~1.5 × 10⁰ | **3.14 × 10⁻²** |
| Biology          | ~1.2 × 10² | ~4.0 × 10⁰  | ~3.0 × 10¹  | ~4.0 × 10¹ | –               |
| Physics          | ~1.0 × 10¹ | ~1.0 × 10⁻³ | ~5.0 × 10⁻² | ~1.0 × 10⁰ | **7.93 × 10⁻⁴** |
| Material Science | ~2.5 × 10¹ | ~3.0 × 10⁰  | ~8.0 × 10⁰  | ~2.5 × 10¹ | –               |

Current results for OpenEvolve are only for two subsets of LSR-Synth. We will update the comprehensive results soon.


------

## 🤝 Contribution

This OpenEvolve example for LLM-SRBench was implemented by [**Haowei Lin**](https://linhaowei1.github.io/) from Peking University. If you encounter any issues or have questions, please feel free to reach out to Haowei via email (linhaowei@pku.edu.cn) for discussion.

