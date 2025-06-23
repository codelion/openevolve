# EVOLVE-BLOCK-START
"""
Optimal-execution example for OpenEvolve
Only the code enclosed by EVOLVE-BLOCK-START / EVOLVE-BLOCK-END
will be mutated by the evolutionary search.
"""
import numpy as np

def create_schedule(volume: float, horizon: int, alpha: float) -> np.ndarray:
    """
    Generate a slice-by-slice schedule.
      alpha < 0  → front–load
      alpha = 0  → uniform
      alpha > 0  → back–load
    """
    weights = np.array([(t + 1) ** alpha for t in range(horizon)], dtype=float)
    weights /= weights.sum()
    return volume * weights

def evaluate_alpha(alpha: float, horizon: int, scenarios: int) -> float:
    """
    Average per-share slippage of an 'alpha' schedule
    over a number of random market scenarios.
    (simulate_execution is defined outside the evolve block.)
    """

    rng = np.random.default_rng()
    cost = 0.0
    for _ in range(scenarios):
        vol  = rng.integers(100, 1000)
        side = rng.choice(["buy", "sell"])
        cost += simulate_execution(volume=vol, side=side, alpha=alpha, rng=rng)
    return cost / scenarios

def search_algorithm(
        iterations: int = 250,
        horizon: int = 10,
        alpha_bounds: tuple = (-1.0, 3.0),
        scenarios: int = 40,
):
    """
    Very simple random search for a good ‘alpha’.
    """
    best_alpha = np.random.uniform(*alpha_bounds)
    best_cost  = evaluate_alpha(best_alpha, horizon, scenarios)

    for _ in range(iterations):
        alpha = np.random.uniform(*alpha_bounds)
        cost  = evaluate_alpha(alpha, horizon, scenarios)
        if cost < best_cost:
            best_alpha, best_cost = alpha, cost

    return best_alpha, best_cost
# EVOLVE-BLOCK-END


# ------------  Fixed (non-evolved) part below -----------------
import numpy as np

def create_schedule(volume: float, horizon: int, alpha: float) -> np.ndarray:
    """
    Duplicate of the schedule helper so the evaluator can import it too.
    (This definition is outside the evolve block and therefore fixed.)
    """
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
    """
    Ultra-light order-book / price-impact simulation.
    Returns *per-share* slippage (positive number, lower is better).
    """
    if rng is None:
        rng = np.random.default_rng()

    mid0      = 100.0               # reference price
    mid_price = mid0
    slices    = create_schedule(volume, horizon, alpha)
    slippage  = 0.0

    for child_vol in slices:
        # mid-price random walk
        mid_price += rng.normal(0.0, 0.05)

        # very simple linear impact model
        impact = (child_vol / depth) * (spread / 2)

        if side == "buy":
            exec_px  = mid_price + spread / 2 + impact
            slippage += (exec_px - mid0) * child_vol
        else:                       # sell
            exec_px  = mid_price - spread / 2 - impact
            slippage += (mid0 - exec_px) * child_vol

    return slippage / volume        # per-share value

def run_search():
    """
    Entry point required by the evaluator.
    Returns the best ‘alpha’ found and the cost on the
    training scenarios used inside search_algorithm.
    """
    alpha, cost = search_algorithm()
    return alpha, cost

if __name__ == "__main__":
    best_alpha, est_cost = run_search()
    print(f"Best alpha: {best_alpha:.3f}  |  Estimated average slippage: {est_cost:.5f}")
