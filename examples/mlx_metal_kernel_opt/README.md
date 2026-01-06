# MLX Metal Kernel Optimization (Qwen3-0.6B-bf16)

This example demonstrates evolutionary optimization of a custom Apple Silicon **Metal** attention kernel using OpenEvolve and MLX’s `metal_kernel` API. The target workload is **Grouped Query Attention (GQA)** for the MLX‑LM model `mlx-community/Qwen3-0.6B-bf16`.

## Target

- **Model**: `mlx-community/Qwen3-0.6B-bf16`
- **Attention**: GQA **16 query heads : 8 KV heads** (2:1), **head_dim=128**, **hidden_size=2048**
- **Dtype**: `bfloat16` (bf16) by default for this model
- **Baseline**: `mx.fast.scaled_dot_product_attention`
- **Hardware**: Apple Silicon (Metal)

## Key files

- `initial_program.py`: starting point (contains `create_metal_qwen3_optimization_hook()` and the EVOLVE‑BLOCK)
- `evaluator.py`: correctness + benchmarking + safety checks for candidates
- `qwen3_benchmark_suite.py`: benchmark definitions and subprocess runner
- `mlx_lm_generate_with_hook.py`: wrapper to apply the attention hook **inside** the `mlx_lm.generate` subprocess
- `run_benchmarks.py`: convenience benchmark runner (baseline vs optimized)
- `config.yaml`: OpenEvolve config and optimization prompt
- `run_evolve_experiment.sh`: convenience script for isolated runs (`output_dir` + `db_path`)

## Important: evaluation validity (before vs after)

Earlier versions of this example could produce misleading “best program” artifacts and invalid performance comparisons. The main issues and the fixes:

| Area | Before | After |
|------|--------|-------|
| **Subprocess benchmark hook** | Benchmarks ran `python -m mlx_lm.generate ...` via `subprocess.run(...)`, so any monkey‑patch in the parent process was **not applied** in the child process (baseline and “optimized” could run the same attention). | Benchmarks can run via `mlx_lm_generate_with_hook.py --hook-program ...` so the patch is applied **inside the subprocess**. |
| **bf16 correctness** | Correctness used `float32` inputs; candidates could pass tests but fail in real bf16 inference (Metal compilation/runtime errors). | Correctness covers **bf16**, and deterministic Metal compilation errors are treated as normal candidate failures. |
| **Architecture alignment** | Docs/prompt/MockArgs assumed **40:8** heads and **hidden_size=5120** (incorrect for Qwen3‑0.6B). | Docs/prompt/MockArgs aligned to **16:8** and **hidden_size=2048**. |

Because of these fixes, we intentionally avoid hard-coded performance claims here. **Rerun the benchmarks on your own machine** and record results in your environment.

## Run evolution

From this directory:

```bash
export OPENAI_API_KEY="..."  # or set GEMINI_API_KEY; see the runner script
bash run_evolve_experiment.sh --foreground
```

This writes a new `openevolve_output_<timestamp>/` directory containing logs, checkpoints, best programs, and an isolated database.

If you prefer running the CLI directly:

```bash
export OPENAI_API_KEY="..."
python -m openevolve.cli ./initial_program.py ./evaluator.py -c ./config.yaml -o ./openevolve_output
```

## Run benchmarks (baseline vs optimized)

To compare the MLX baseline against the best evolved program:

```bash
python run_benchmarks.py --mode compare --model mlx-community/Qwen3-0.6B-bf16 --output-dir results
```

## How to verify the validity fixes are active

When the hook is enabled, the optimized path should execute via the wrapper:

- `mlx_lm_generate_with_hook.py --hook-program <best_program.py> --model ...`

You can also sanity-check that correctness is exercising bf16 by running evolution on a machine where bf16 Metal compilation errors are expected for invalid kernels: such candidates should be rejected early by correctness gating rather than becoming “best programs”.

## Limitations & potential improvements (follow-up work)

This example intentionally uses **end-to-end generation benchmarks** (`mlx_lm.generate`) to measure real workloads, but that comes with trade-offs:

- **Benchmark noise & overhead**: subprocess startup, model loading, and generation variability can dwarf small kernel deltas (especially for short prompts). A complementary **microbenchmark** that times only the attention kernel would provide a cleaner signal.
- **Serial evaluation by default**: candidates are evaluated sequentially (`parallel_evaluations: 1`) to keep GPU memory predictable. More parallelism may be possible with careful isolation, but it needs engineering.
- **Compile-time dominates early search**: bf16 compilation failures are common and deterministic; caching compilation outcomes or factoring compilation into a cheaper gating stage may speed up evolution.

We plan to open follow-up issues to track improvements to the benchmarking/evolution signal and workflow.