#!/usr/bin/env python3
"""
Run MLX-LM generation with a custom attention monkey-patch applied *inside* the
current process.

Why this exists
---------------
Many benchmarking utilities run `mlx_lm.generate` via `subprocess.run(...)`.
Any monkey-patch done in the parent process (e.g. replacing
`mlx_lm.models.qwen3.Attention`) does NOT propagate into the child process.

This wrapper makes the patch effective by:
1) loading an evolved program file (e.g. best_program.py)
2) calling its `create_metal_qwen3_optimization_hook()` to apply the patch
3) running `mlx_lm.generate` in the same process (via `runpy`)
"""

from __future__ import annotations

import argparse
import importlib.util
import runpy
import sys
from types import ModuleType
from typing import List, Optional, Tuple, Any


def _load_module_from_path(module_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("openevolve_mlx_metal_hook_program", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load hook program from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _apply_hook_from_program(module_path: str) -> Tuple[Any, Any]:
    program = _load_module_from_path(module_path)

    hook_factory = getattr(program, "create_metal_qwen3_optimization_hook", None)
    if hook_factory is None:
        raise RuntimeError(
            "Hook factory `create_metal_qwen3_optimization_hook()` not found in hook program."
        )

    apply_hook, remove_hook = hook_factory()
    original_attention = apply_hook()
    if original_attention is None:
        raise RuntimeError("Failed to apply optimization hook (original_attention is None).")

    return original_attention, remove_hook


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run `mlx_lm.generate` with a custom attention hook applied in-process."
    )
    parser.add_argument("--hook-program", required=True, help="Path to evolved program (e.g. best_program.py)")
    parser.add_argument("--model", required=True, help="Model id/path to pass to mlx_lm.generate")
    parser.add_argument("--prompt", required=True, help="Prompt string")
    parser.add_argument("--max-tokens", required=True, type=int, help="Max tokens to generate")

    args = parser.parse_args(argv)

    original_attention = None
    remove_hook = None
    try:
        original_attention, remove_hook = _apply_hook_from_program(args.hook_program)

        # Mimic `python -m mlx_lm.generate ...`
        sys.argv = [
            "mlx_lm.generate",
            "--model",
            args.model,
            "--prompt",
            args.prompt,
            "--max-tokens",
            str(args.max_tokens),
        ]

        try:
            runpy.run_module("mlx_lm.generate", run_name="__main__")
            return 0
        except SystemExit as e:
            # Preserve the exit code from mlx_lm.generate
            code = e.code
            if code is None:
                return 0
            if isinstance(code, int):
                return code
            return 1

    finally:
        if remove_hook is not None and original_attention is not None:
            try:
                remove_hook(original_attention)
            except Exception:
                # Non-fatal in a one-shot subprocess wrapper
                pass


if __name__ == "__main__":
    raise SystemExit(main())


