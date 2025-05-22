# Evolving Symbolic Regression with OpenEvolve on LLM-SRBench 🧬🔍

This example demonstrates how **OpenEvolve** can be utilized to perform **symbolic regression** tasks using the **[LLM-SRBench benchmark](https://arxiv.org/pdf/2504.10415)**. It showcases OpenEvolve's capability to evolve Python code, transforming simple mathematical expressions into more complex and accurate models that fit given datasets.

------

## 🎯 Problem Description: Symbolic Regression on LLM-SRBench

**Symbolic Regression** is the task of discovering a mathematical expression that best fits a given dataset. Unlike traditional regression techniques that optimize parameters for a predefined model structure, symbolic regression aims to find both the **structure of the model** and its **parameters**.

This example leverages **LLM-SRBench**, a benchmark specifically designed for Large Language Model-based Symbolic Regression. The core objective is to use OpenEvolve to evolve an initial, often simple, model (e.g., a linear model) into a more sophisticated symbolic expression. This evolved expression should accurately capture the underlying relationships within various scientific datasets provided by the benchmark.

------

## 🚀 Getting Started

Follow these steps to set up and run the symbolic regression benchmark example:

### 1. Configure API Secrets

You'll need to provide your API credentials for the language models used by OpenEvolve.

- Create a `secrets.yaml` file in the example directory.
- Add your API key and model preferences:

YAML

```
# secrets.yaml
api_key: <YOUR_OPENAI_API_KEY>
api_base: "https://api.openai.com/v1"  # Or your custom endpoint
primary_model: "gpt-4o"
secondary_model: "o3" # Or another preferred model for specific tasks
```

Replace `<YOUR_OPENAI_API_KEY>` with your actual OpenAI API key.

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

OpenEvolve will iteratively modify the Python code within the `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` markers in `initial_program.py`. The goal is to transform the simple initial model into a more complex and accurate symbolic expression that minimizes the Mean Squared Error (MSE) on the training data.

An evolved `func` might, for instance, discover a non-linear expression like:

```python
# Hypothetical example of what OpenEvolve might find:
def func(x, params):
   # Assuming X_train_scaled maps to x and const maps to a parameter in params
   predictions = np.sin(x[:, 0]) * x[:, 1]**2 + params[0]
   return predictions
```

*(This is a simplified, hypothetical example to illustrate the transformation.)*

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

| **Task Category**       | Med. NMSE (Test) | Med. R2 (Test) | **Med. NMSE (OOD Test)** | **Med. R2 (OOD Test)** |
| ----------------------- | ---------------- | -------------- | ------------------------ | ---------------------- |
| Chemistry (36 tasks)    | 2.3419e-06       | 1.000          | 3.1384e-02               | 0.9686                 |
| Physics (44 tasks)      | 1.8548e-05       | 1.000          | 7.9255e-04               | 0.9992                 |

Current results are only for two subset of LSR-Synth. We will update the comprehensive results soon.

------

## 🤝 Contribution

This OpenEvolve example for LLM-SRBench was implemented by [**Haowei Lin**](https://linhaowei1.github.io/) from Peking University. If you encounter any issues or have questions, please feel free to reach out to Haowei via email (linhaowei@pku.edu.cn) for discussion.

