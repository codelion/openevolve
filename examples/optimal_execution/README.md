README  
======  
   
Optimal-Execution Toy Benchmark for OpenEvolve  
---------------------------------------------  
   
This repository contains a **minimal yet complete** benchmark that lets an evolutionary-search engine learn how to execute a fixed quantity of shares in an order-book with market impact.    
It mirrors the structure of the earlier “function-minimisation” example but replaces the mathematical objective with a *trading* objective:  
   
*Minimise implementation-shortfall / slippage when buying or selling a random volume during a short horizon.*  
   
The benchmark is intentionally lightweight – short Python, no external dependencies – yet it shows every building-block you would find in a realistic execution engine:  
   
1. synthetic order-book generation    
2. execution-schedule parameterisation    
3. a search / learning loop confined to an `EVOLVE-BLOCK`    
4. an **independent evaluator** that scores candidates on unseen market scenarios.  
   
-------------------------------------------------------------------------------  
   
Repository Layout  
-----------------  
   
```  
.  
├── initial_program.py   # candidate – contains the EVOLVE-BLOCK  
├── evaluator.py         # ground-truth evaluator  
└── README.md            # ← you are here  
```  
   
Why two files?  
• `initial_program.py` is what the evolutionary framework mutates.    
• `evaluator.py` is trusted, *never* mutated and imports nothing except the  
  candidate’s public `run_search()` function.  
   
-------------------------------------------------------------------------------  
   
Quick-start  
-----------  
   
```  
python initial_program.py  
    # Runs the candidate’s own training loop (random-search on α)  
   
python evaluator.py initial_program.py  
    # Scores the candidate on fresh market scenarios  
```  
   
Typical console output:  
   
```  
Best alpha: 1.482  |  Estimated average slippage: 0.00834  
{'value_score': 0.213, 'speed_score': 0.667,  
 'reliability': 1.0, 'overall_score': 0.269}  
```  
   
-------------------------------------------------------------------------------  
   
1. Mechanics – Inside the Candidate (`initial_program.py`)  
----------------------------------------------------------  
   
The file is split into two parts:  
   
### 1.1 EVOLVE-BLOCK (mutable)  
   
```python  
# EVOLVE-BLOCK-START … EVOLVE-BLOCK-END  
```  
   
Only the code between those delimiters will be altered by OpenEvolve.  
Everything else is *frozen*; it plays the role of a “library.”  
   
Current strategy:  
   
1. **Parameter** – a single scalar `alpha (α)`    
   • α < 0  → front-loads the schedule    
   • α = 0  → uniform (TWAP)    
   • α > 0  → back-loads the schedule  
   
2. **Search** – naïve random search over α    
   (`search_algorithm()` evaluates ~250 random α’s and keeps the best.)  
   
3. **Fitness** – measured by `evaluate_alpha()` which, in turn, calls the  
   **fixed** simulator (`simulate_execution`) for many random scenarios and  
   averages per-share slippage.  
   
Return signature required by the evaluator:  
   
```python  
def run_search() -> tuple[float, float]:  
    return best_alpha, estimated_cost  
```  
   
The first element (α) is mandatory; anything after that is ignored by the  
evaluator but can be useful for debugging.  
   
### 1.2 Fixed “library” code (non-mutable)  
   
* `create_schedule(volume, horizon, alpha)`    
  Weights each slice `(t+1)^α`, then normalises to equal volume.  
   
* `simulate_execution(...)`    
  Ultra-simplified micro-structure:  
  
  • The mid-price `P_t` follows a Gaussian random walk    
  • The current spread is constant (`±spread/2`)    
  • Market impact grows linearly with child-order size relative to  
    book depth:    
    `impact = (size / depth) * spread/2`  
  
  Execution price for each slice:  
  
  ```  
  BUY : P_t + spread/2 + impact  
  SELL: P_t - spread/2 - impact  
  ```  
  
  Slippage is summed over the horizon and returned *per share*.  
   
-------------------------------------------------------------------------------  
   
2. Mechanics – The Evaluator (`evaluator.py`)  
---------------------------------------------  
   
The evaluator is the **oracle**; it owns the test scenarios and the scoring  
function.  A successful candidate must *generalise*: the random numbers in  
the evaluator are independent from those inside the candidate.  
   
### 2.1 Process flow  
   
For each of `NUM_TRIALS = 10`:  
   
1. Draw a *fresh* `(volume, side)` pair    
   `volume ∈ [100, 1000]`, `side ∈ {buy, sell}`  
   
2. Call `run_search()` **once** (time-limited to 8 s)  
   
3. Extract α and compute:    
  
   ```  
   cost_candidate = simulate_execution(vol, side, α)  
   cost_baseline  = simulate_execution(vol, side, 0.0)   # uniform TWAP  
   improvement    = (cost_baseline - cost_candidate)  
                    / max(cost_baseline, 1e-9)  
   ```  
   
4. Store runtime and improvement.  
   
### 2.2 Scores  
   
After the 10 trials:  
   
```  
value_score       = mean(max(0, improvement))     ∈ [0, 1]  
speed_score       = min(10, 1/mean(runtime)) / 10 ∈ [0, 1]  
reliability_score = success / 10                  ∈ [0, 1]  
   
overall_score = 0.8·value + 0.1·speed + 0.1·reliability  
```  
   
Intuition:  
   
* **Value** (quality of execution) dominates.  
* **Speed** rewards fast optimisation but is capped.  
* **Reliability** ensures the candidate rarely crashes or times-out.  
   
### 2.3 Stage-based evaluation (optional)  
   
* `evaluate_stage1()` – smoke-test; passes if `overall_score > 0.05`  
* `evaluate_stage2()` – identical to `evaluate()`  
   
Those mirrors the two-stage funnel from the previous demo.  
   
-------------------------------------------------------------------------------  
   
3. Extending the Benchmark  
--------------------------  
   
The framework is deliberately tiny so you can experiment.  
   
Ideas:  
   
1. **Richer parameterisation**    
   • Add `beta` for *U-shape* schedule    
   • Add *child-order participation cap* (%ADV)  
   
2. **Better search / learning**    
   • Replace random search with gradient-free CMA-ES, Bayesian optimisation or  
     even RL inside the EVOLVE-BLOCK.  
   
3. **Enhanced market model**    
   • Stochastic spread    
   • Non-linear impact (`impact ∝ volume^γ`)    
   • Resilience (price reverts after child order)  
   
4. **Multi-objective scoring**    
   Mix risk metrics (variance of slippage) into the evaluator.  
   
When you add knobs, remember:  
   
* All **simulation logic for evaluation must live in `evaluator.py`**.    
  Candidates cannot peek or tamper with it.  
* The evaluator must still be able to extract the *decision variables* from  
  the tuple returned by `run_search()`.  
   
-------------------------------------------------------------------------------  
   
4. Known Limitations  
--------------------  
   
1. **Impact model is linear & memory-less**    
   Good for demonstration; unrealistic for real-world HFT.  
   
2. **No order-book micro-structure**    
   We do not simulate queue positions, cancellations, hidden liquidity, etc.  
   
3. **Single parameter α**    
   Optimal execution in reality depends on volatility, spread forecast,  
   order-book imbalance and so forth.  Here we sidestep all that for clarity.  
   
4. **Random search baseline**    
   Evolutionary engines will easily outperform it; that is the point – we  
   want a hill to climb.  
   
-------------------------------------------------------------------------------  
   
5. FAQ  
------  
Q: **How do I run the example?**
A: Run `python openevolve-run.py examples/optimal_execution/initial_program.py examples/optimal_execution/evaluator.py --iterations 20 --config config.yaml'
Q: **Why does the evaluator re-implement `simulate_execution`?**    
A: To guarantee the candidate cannot cheat by hard-coding answers from its own  
RNG realisations.  
   
Q: **What happens if my `run_search()` returns something weird?**    
A: The evaluator casts the *first* item to `float`.  Non-numeric or `NaN`  
values yield zero score.  
   
Q: **Is it okay to import heavy libraries (pandas, torch) inside the EVOLVE-BLOCK?**    
A: Technically yes, but remember the 8-second time-out and the judge’s machine  
may not have GPU or large RAM.  
   
-------------------------------------------------------------------------------  
   
6. License  
----------  
   
The example is released under the MIT License – do whatever you like, but  
please keep references to the original authorship when redistributing.  
   
