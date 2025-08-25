from argparse import ArgumentParser

from openevolve import OpenEvolve
from evaluator import EvaluationObject

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--problem",
        help="Which problem to solve",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        help="Timeout for a single submission (in seconds)",
        type=int,
        default=60,
    )

    args = parser.parse_args()
    eval_obj = EvaluationObject(args.problem, args.timeout)
    evolve = OpenEvolve("initial_program.py", "", eval_obj, "config.yaml")