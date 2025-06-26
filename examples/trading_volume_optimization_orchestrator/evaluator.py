"""
Evaluator for the optimal-execution example.
The only requirement for the candidate program is a `run_search`
function that returns *at least* the first element `alpha`.
"""
import importlib.util
import time
import concurrent.futures
import traceback
import numpy as np
import sys

# -----------------------------------------------------------------
# Small helper copied from the previous demo
def run_with_timeout(func, args=(), kwargs=None, timeout_seconds=8):
    if kwargs is None:
        kwargs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(func, *args, **kwargs)
        try:
            return fut.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Timed-out after {timeout_seconds}s")

        # -----------------------------------------------------------------
# A *fixed* copy of the market simulator – identical to the one in
# initial_program.py but held here so candidates cannot tamper with it.
def create_schedule(volume: float, horizon: int, alpha: float) -> np.ndarray:
    weights = np.array([(t + 1) ** alpha for t in range(horizon)], dtype=float)
    weights /= weights.sum()
    return volume * weights

def simulate_execution(
        volume: float,
        side: str,
        alpha: float,
        horizon: int = 10,
        spread: float = 0.02,
        depth: float = 1_000.0,
        rng: np.random.Generator | None = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng()

    mid0      = 100.0
    mid_price = mid0
    slices    = create_schedule(volume, horizon, alpha)
    slippage  = 0.0

    for child_vol in slices:
        mid_price += rng.normal(0.0, 0.05)
        impact = (child_vol / depth) * (spread / 2)
        if side == "buy":
            exec_px  = mid_price + spread / 2 + impact
            slippage += (exec_px - mid0) * child_vol
        else:
            exec_px  = mid_price - spread / 2 - impact
            slippage += (mid0 - exec_px) * child_vol
    return slippage / volume

# -----------------------------------------------------------------
def evaluate(program_path: str):
    """
    Score a candidate program on 10 fresh, unseen market scenarios.
    Metric: improvement relative to a naïve uniform schedule (alpha = 0).
    """
    try:
        spec = importlib.util.spec_from_file_location("candidate", program_path)
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)

        if not hasattr(candidate, "run_search"):
            return {"overall_score": 0.0, "error": "Missing run_search()"}

        NUM_TRIALS = 10
        rng        = np.random.default_rng(seed=42)

        improvements = []
        times        = []
        success      = 0

        for trial in range(NUM_TRIALS):
            volume = rng.integers(100, 1000)
            side   = rng.choice(["buy", "sell"])

            try:
                t0      = time.time()
                result  = run_with_timeout(candidate.run_search, timeout_seconds=8)
                t1      = time.time()
                times.append(t1 - t0)

                # Extract alpha (first value) – additional items ignored
                if isinstance(result, (tuple, list)):
                    alpha = float(result[0])
                else:
                    alpha = float(result)

                    # Our simulator – independent of candidate’s own one
                cost_candidate = simulate_execution(volume, side, alpha, rng=rng)
                cost_baseline  = simulate_execution(volume, side, 0.0,  rng=rng)

                improvement = (cost_baseline - cost_candidate) / max(cost_baseline, 1e-9)
                improvements.append(improvement)
                success += 1

            except TimeoutError as e:
                print(f"Trial {trial}: {e}")
            except Exception as e:
                print(f"Trial {trial}: {e}\n{traceback.format_exc()}")

        if success == 0:
            return {"overall_score": 0.0, "error": "All trials failed"}

        avg_improvement = float(np.mean(improvements))
        avg_time        = float(np.mean(times))

        value_score      = max(0.0, avg_improvement)           # already 0-1 range
        speed_score      = min(10.0, 1.0 / avg_time) / 10.0    # cap influence
        reliability_score= success / NUM_TRIALS

        overall_score = 0.8 * value_score + 0.1 * speed_score + 0.1 * reliability_score

        return {
            "value_score"     : value_score,
            "speed_score"     : speed_score,
            "reliability"     : reliability_score,
            "overall_score"   : overall_score,
        }
    except Exception as e:
        print(traceback.format_exc())
        return {"overall_score": 0.0, "error": str(e)}

    # -----------------------------------------------------------------
# Two quick stage helpers (optional, mirrors the original demo)

def evaluate_stage1(program_path: str):
    """
    Smoke-test: does it run & make *some* improvement?
    """
    res = evaluate(program_path)
    ok  = res.get("overall_score", 0.0) > 0.05
    return {"runs_successfully": 1.0 if ok else 0.0, **res}

def evaluate_stage2(program_path: str):
    return evaluate(program_path)

# -----------------------------------------------------------------
if __name__ == "__main__":
    # Allow quick manual test:  python evaluator.py initial_program.py
    if len(sys.argv) != 2:
        print("Usage: python evaluator.py path/to/initial_program.py")
        sys.exit(0)
    scores = evaluate(sys.argv[1])
    print(scores)
