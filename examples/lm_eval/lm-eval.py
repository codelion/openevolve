"""
OpenEvolve <-> lm-evaluation-harness adapter

Implements generation only, no loglikelihood. Tasks such as GSM8K / BoolQ / MMLU-Math /
AQUA-RAT and most code suites should work fine because they grade on the generated
answer string.
"""

from __future__ import annotations
import subprocess, tempfile, json, os, argparse, math, pathlib
from pathlib import Path
from typing import List, Dict, Tuple, Any, Iterable

import lm_eval
from lm_eval.tasks import TaskManager
from lm_eval.evaluator import evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from datetime import datetime

# cd to the parent parent directory of this file
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PIPELINE_CMD = ["python3", "openevolve-run.py"]


@register_model("openevolve")
class OpenEvolve(LM):
    def __init__(
        self,
        init_file: str = "initial_content_stub.txt",
        evaluator_file: str = "evaluator_stub.py",
        config_file: str = "config.yml",
        iterations: int = 5,
        extra_param: List[str] = [],
        **kwargs,
    ):
        super().__init__()
        self.init_file = init_file
        self.evaluator_file = evaluator_file
        self.iterations = iterations
        self.extra_param = extra_param
        self.config_file = config_file

        # folder must match prompt:template_dir in config.yml!
        self.prompt_path = "examples/lm_eval/prompts/system_message.txt"
        self.evaluator_prompt_path = "examples/lm_eval/prompts/evaluator_system_message.txt"
        self.best_path = "examples/lm_eval/openevolve_output/best/best_program.txt"
        self.base_system_message = "You are an expert task solver, with a lot of commonsense, math, language and coding knowledge.\n\nConsider this task:\n```{prompt}´´´"

    def generate(self, prompts: List[str], max_gen_toks: int = None, stop=None, **kwargs):
        outs = []
        for prompt in prompts:
            # Task prompt becomes the system message. User prompt is the evolutionary logic.
            # We create temporary prompt files with the system message
            with Path(self.prompt_path).open("w") as f:
                f.write(self.base_system_message.format(prompt=prompt))

            with Path(self.evaluator_prompt_path).open("w") as f:
                f.write(self.base_system_message.format(prompt=prompt))

            cmd = (
                PIPELINE_CMD
                + ["--config", self.config_file]
                + ["--iterations", str(self.iterations)]
                + self.extra_param
                + [self.init_file, self.evaluator_file]
            )
            print(f"Running command: {' '.join(cmd)}")
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                text = res.stdout.strip()
                print(f"Process output: {text}")
            except subprocess.CalledProcessError as e:
                print(f"Command failed with return code {e.returncode}")
                print(f"stderr: {e.stderr}")
                text = ""

            print(f"# Prompt: {prompt}")
            with Path(self.best_path).open("r") as f:
                best = f.read().strip()
                print(f"# Answer: {best}")

            # honour stop tokens
            if stop:
                for s in stop:
                    idx = best.find(s)
                    if idx != -1:
                        best = best[:idx]
                        break
            outs.append(best)
        return outs

    # for tasks that ask for log likelihood, indicate that it is unsupported
    def loglikelihood(self, requests: Iterable[Tuple[str, str]], **kw):
        # return [(-math.inf, False) for _ in requests]
        raise NotImplementedError

    def loglikelihood_rolling(self, requests: Iterable[str], **kw):
        # return [(-math.inf, False) for _ in requests]
        raise NotImplementedError

    def generate_until(self, requests: Iterable[Any], **kw) -> List[str]:
        ctxs, stops = [], []

        for req in requests:
            # ---------------- old: plain tuple ----------------
            if isinstance(req, tuple):
                ctx, until = req

            # -------------- new: Instance object --------------
            else:
                ctx = req.args[0]  # first positional arg
                until = []
                # if a second positional arg exists and is list-like,
                # treat it as the stop sequence
                if len(req.args) > 1 and isinstance(req.args[1], (list, tuple)):
                    until = list(req.args[1])

            ctxs.append(ctx)
            stops.append(until)

        # 2) run your real generator once per context
        gens = self.generate(ctxs, stop=None)

        # 3) post-trim at the first stop sequence
        cleaned = []
        for g, until in zip(gens, stops):
            for s in until:
                idx = g.find(s)
                if idx != -1:
                    g = g[:idx]
                    break
            cleaned.append(g)
        return cleaned


if __name__ == "__main__":
    # cli arguments for primary model, secondary model, iterations, config and tasks
    p = argparse.ArgumentParser(
        description="OpenEvolve <-> lm-evaluation-harness adapter.",
    )
    p.add_argument("--config", default="examples/lm_eval/config.yml", help="config file")
    p.add_argument(
        "--init_file",
        default="examples/lm_eval/initial_content_stub.txt",
        help="initial content file",
    )
    p.add_argument(
        "--evaluator_file", default="examples/lm_eval/evaluator_stub.py", help="evaluator file"
    )
    p.add_argument("--iterations", default=5, type=int, help="number of iterations")
    p.add_argument(
        "--limit",
        default=None,
        type=int,
        help="limit the number of examples per task that are executed",
    )
    # p.add_argument("--tasks", default="boolq,gsm8k,mmlu", help="comma-list of tasks to evaluate")
    p.add_argument("--tasks", default="gsm8k", help="list of tasks to evaluate")
    p.add_argument("--output_path", default="results", help="output path for results")
    args = p.parse_args()

    lm_obj = OpenEvolve(
        init_file=args.init_file,
        evaluator_file=args.evaluator_file,
        iterations=args.iterations,
        config_file=args.config,
    )

    task_dict = lm_eval.tasks.get_task_dict(args.tasks.split(","))

    results = evaluate(
        lm=lm_obj,
        task_dict=task_dict,
        limit=args.limit,
    )

    # write out the results
    pathlib.Path(
        args.output_path,
    ).mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = pathlib.Path(
        os.path.join(
            args.output_path,
            f"{timestamp}_iter{args.iterations}.json",
        )
    )

    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    # print result summary
    short = {}
    for task, metrics in results["results"].items():
        # pick the first value that is a real number
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                short[task] = (key, val)  # store *both* name & value
                break

    print(f"Full results written to {results_path}\n")
    print("Headline metrics:")
    for task, (name, value) in short.items():
        print(f"  {task:<15} {name:<12} {value:.3%}")

    print("\nNote: Never cite the overall average when some components were skipped!")
