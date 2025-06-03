# lm-eval.py

`lm-eval.py` provides basic benchmark capability for LLM feedback-based evolutionary task solving. The benchmark framework is [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

*Limitation:* Only generation-only tasks such as gsm8k are supported. This is because tasks that require loglikelihood probabilities are not well applicable to agents.

## Usage

```bash
$ python3 examples/lm_eval/lm-eval.py -h
usage: lm-eval.py [-h] [--config CONFIG] [--init_file INIT_FILE] [--evaluator_file EVALUATOR_FILE] [--iterations ITERATIONS] [--limit LIMIT] [--tasks TASKS]
                  [--output_path OUTPUT_PATH]

OpenEvolve <-> lm-evaluation-harness adapter.

options:
  -h, --help            show this help message and exit
  --config CONFIG       config file
  --init_file INIT_FILE
                        initial content file
  --evaluator_file EVALUATOR_FILE
                        evaluator file
  --iterations ITERATIONS
                        number of iterations
  --limit LIMIT         limit the number of examples per task that are executed
  --tasks TASKS         list of tasks to evaluate
  --output_path OUTPUT_PATH
                        output path for results
```

Early examples that **were meant to** indicate that more evolution iterations improve task performance -- I suspect the prompting may not be ideal yet:
```
$ python3 examples/lm_eval/lm-eval.py --tasks gsm8k --limit 10 --iterations 1
[..]
Headline metrics:
  gsm8k           exact_match,strict-match 80.000%
[..]


$ python3 examples/lm_eval/lm-eval.py --tasks gsm8k --limit 10 --iterations 3
[..]
Headline metrics:
  gsm8k           exact_match,strict-match 90.000%
[..]

$ python3 examples/lm_eval/lm-eval.py --tasks gsm8k --limit 10 --iterations 10
[..]
Headline metrics:
  gsm8k           exact_match,strict-match 80.000%
[..]

$ python3 examples/lm_eval/lm-eval.py --tasks gsm8k --limit 10 --iterations 15
[..]
Headline metrics:
  gsm8k           exact_match,strict-match 70.000%
[..]
```

## Warning

- Be aware that this is an early implementation. No extensive benchmarks have been executed so far. With a limit to 10 tasks and 10 iterations, the benchmark is meaningless as is.
- Use the --limit parameter only for tests, not for metric generation.
- Do not cite the metrics that result from the script execution blindly without reviewing the solution first.

## References

```bibtex
@misc{eval-harness,
    author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
    title        = {The Language Model Evaluation Harness},
    month        = 07,
    year         = 2024,
    publisher    = {Zenodo},
    version      = {v0.4.3},
    doi          = {10.5281/zenodo.12608602},
    url          = {https://zenodo.org/records/12608602}
}
```