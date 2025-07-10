# Web Scraper Evolution with optillm

This example demonstrates how to use [optillm](https://github.com/codelion/optillm) with OpenEvolve to leverage test-time compute techniques for improved code evolution accuracy. We'll evolve a web scraper that extracts structured data from documentation pages, showcasing two key optillm features:

1. **readurls plugin**: Automatically fetches webpage content when URLs are mentioned in prompts
2. **Inference optimization**: Uses techniques like Mixture of Agents (MoA) to improve response accuracy

## Why optillm?

Traditional LLM usage in code evolution has limitations:
- LLMs may not have knowledge of the latest library documentation
- Single LLM calls can produce inconsistent or incorrect code
- No ability to dynamically fetch relevant documentation during evolution

optillm solves these problems by:
- **Dynamic Documentation Fetching**: The readurls plugin automatically fetches and includes webpage content when URLs are detected in prompts
- **Test-Time Compute**: Techniques like MoA generate multiple responses and synthesize the best solution
- **Flexible Routing**: Can route requests to different models based on requirements

## Problem Description

We're evolving a web scraper that extracts API documentation from Python library documentation pages. The scraper needs to:
1. Parse HTML documentation pages
2. Extract function signatures, descriptions, and parameters
3. Structure the data in a consistent format
4. Handle various documentation formats

This is an ideal problem for optillm because:
- The LLM benefits from seeing actual documentation HTML structure
- Accuracy is crucial for correct parsing
- Different documentation sites have different formats

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   OpenEvolve    │────▶│     optillm     │────▶│   Local LLM     │
│                 │     │  (proxy:8000)   │     │  (Qwen-0.5B)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ├── readurls plugin
                               │   (fetches web content)
                               │
                               └── MoA optimization
                                   (improves accuracy)
```

## Setup Instructions

### 1. Install and Configure optillm

```bash
# Clone optillm
git clone https://github.com/codelion/optillm.git
cd optillm

# Install dependencies
pip install -r requirements.txt

# Start optillm proxy with local inference server (in a separate terminal)
export OPTILLM_API_KEY=optillm
python optillm.py --port 8000
```

optillm will now be running on `http://localhost:8000` with its built-in local inference server.

**Note for Non-Mac Users**: This example uses `Qwen/Qwen3-1.7B-MLX-bf16` which is optimized for Apple Silicon (M1/M2/M3 chips). If you're not using a Mac, you should:

1. **For NVIDIA GPUs**: Use a CUDA-compatible model like:
   - `Qwen/Qwen2.5-32B-Instruct` (best quality, high VRAM)
   - `Qwen/Qwen2.5-14B-Instruct` (good balance)
   - `meta-llama/Llama-3.1-8B-Instruct` (efficient option)
   - `Qwen/Qwen2.5-7B-Instruct` (lower VRAM)

2. **For CPU-only**: Use a smaller model like:
   - `Qwen/Qwen2.5-7B-Instruct` (7B parameters)
   - `meta-llama/Llama-3.2-3B-Instruct` (3B parameters)
   - `Qwen/Qwen2.5-3B-Instruct` (3B parameters)

3. **Update the config**: Replace the model names in `config.yaml` with your chosen model:
   ```yaml
   models:
     - name: "readurls-your-chosen-model"
       weight: 0.9
     - name: "moa&readurls-your-chosen-model"
       weight: 0.1
   ```

### 2. Install Web Scraping Dependencies

```bash
# Install required Python packages for the example
pip install -r examples/web_scraper_optillm/requirements.txt
```

### 3. Run the Evolution

```bash
# From the openevolve root directory
export OPENAI_API_KEY=optillm
python openevolve-run.py examples/web_scraper_optillm/initial_program.py \
    examples/web_scraper_optillm/evaluator.py \
    --config examples/web_scraper_optillm/config.yaml \
    --iterations 100
```

The configuration demonstrates both optillm capabilities:
- **Primary model (90%)**: `readurls-Qwen/Qwen3-1.7B-MLX-bf16` - fetches URLs mentioned in prompts
- **Secondary model (10%)**: `moa&readurls-Qwen/Qwen3-1.7B-MLX-bf16` - uses Mixture of Agents for improved accuracy

## How It Works

### 1. readurls Plugin

When the evolution prompt contains URLs (e.g., "Parse the documentation at https://docs.python.org/3/library/json.html"), the readurls plugin:
1. Detects the URL in the prompt
2. Fetches the webpage content
3. Extracts text and table data
4. Appends it to the prompt as context

This ensures the LLM has access to the latest documentation structure when generating code.

### 2. Mixture of Agents (MoA)

The MoA technique improves accuracy by:
1. Generating 3 different solutions to the problem
2. Having each "agent" critique all solutions
3. Synthesizing a final, improved solution based on the critiques

This is particularly valuable for complex parsing logic where multiple approaches might be valid.

### 3. Evolution Process

1. **Initial Program**: A basic BeautifulSoup scraper that extracts simple text
2. **Evaluator**: Tests the scraper against real documentation pages, checking:
   - Correct extraction of function names
   - Accurate parameter parsing
   - Proper handling of edge cases
3. **Evolution**: The LLM improves the scraper by:
   - Fetching actual documentation HTML (via readurls)
   - Generating multiple parsing strategies (via MoA)
   - Learning from evaluation feedback

## Actual Evolution Results

Based on our evolution run, here's what we achieved:

### Performance Metrics
- **Initial Score**: 0.6864 (72.2% accuracy, 32.5% completeness)
- **Final Score**: 0.7458 (83.3% accuracy, 37.5% completeness)
- **Improvement**: +8.6% overall performance (+11.1% accuracy)
- **Time to Best**: Found optimal solution by iteration 3 (within 10 minutes)

### Key Evolution Improvements

**Initial Program** (Basic approach):
```python
# Simple code block parsing
code_blocks = soup.find_all('code')
for block in code_blocks:
    text = block.get_text(strip=True)
    if '(' in text and ')' in text:
        # Extract function info
```

**Evolved Program** (Sophisticated multi-strategy parsing):
```python
# 1. Code blocks
code_blocks = soup.find_all('code')
# 2. Headers (h3)
h3_blocks = soup.find_all('h3')
# 3. Documentation signatures
dt_blocks = soup.find_all('dt', class_='sig')
# 4. Table-based documentation (NEW!)
table_blocks = soup.find_all('table')
for block in table_blocks:
    rows = block.find_all('tr')
    for row in rows:
        cells = row.find_all('td')
        if len(cells) >= 2:
            signature = cells[0].get_text(strip=True)
            description = cells[1].get_text(strip=True)
            # Extract structured function data
```

### What optillm Contributed

1. **Early Discovery**: Found best solution by iteration 3, suggesting enhanced reasoning helped quickly identify effective parsing strategies
2. **Table Parsing Innovation**: The evolved program added sophisticated table parsing logic that wasn't in the initial version
3. **Robust Architecture**: Multiple fallback strategies ensure the scraper works across different documentation formats

## Monitoring Progress

Watch the evolution progress and see how optillm enhances the process:

```bash
# View optillm logs (in the terminal running optillm)
# You'll see:
# - URLs being fetched by readurls
# - Multiple completions generated by MoA
# - Final synthesized responses

# View OpenEvolve logs
tail -f examples/web_scraper_optillm/openevolve_output/evolution.log
```

## Results Analysis

After 100 iterations of evolution, here's what we achieved:

### Quantitative Results
- **Accuracy**: 72.2% → 83.3% (+11.1% improvement)
- **Completeness**: 32.5% → 37.5% (+5% improvement) 
- **Robustness**: 100% (maintained - no parsing errors)
- **Combined Score**: 0.6864 → 0.7458 (+8.6% improvement)

### Qualitative Improvements
1. **Multi-Strategy Parsing**: Added table-based extraction for broader documentation format support
2. **Robust Function Detection**: Improved pattern matching for function signatures
3. **Better Parameter Extraction**: Enhanced parameter parsing from various HTML structures
4. **Error Resilience**: Maintained 100% robustness with no parsing failures

### Evolution Pattern
- **Early Success**: Best solution found by iteration 3 (within 10 minutes)
- **Plateau Effect**: Algorithm maintained optimal score from iteration 3-90
- **Island Migration**: MAP-Elites explored alternatives but local optimum was strong

Compare the evolution:
```bash
# View the final evolved program
cat examples/web_scraper_optillm/openevolve_output/best/best_program.py

# Compare initial vs final
diff examples/web_scraper_optillm/initial_program.py \
     examples/web_scraper_optillm/openevolve_output/best/best_program.py
```

## Key Insights from This Run

1. **optillm Enhanced Early Discovery**: The best solution was found by iteration 3, suggesting optillm's test-time compute (MoA) and documentation access (readurls) helped quickly identify effective parsing strategies.

2. **Smaller Models Can Excel**: The 1.7B Qwen model with optillm achieved significant improvements (+8.6%), proving that test-time compute can make smaller models highly effective.

3. **Local Optimization Works**: Fast inference times (<100ms after initial) show that local models with optillm provide both efficiency and quality.

4. **Pattern: Quick Discovery, Then Plateau**: Evolution found a strong local optimum quickly. This suggests the current test cases were well-solved by the table parsing innovation.

5. **optillm Plugin Value**: The evolved program's sophisticated multi-strategy approach (especially table parsing) likely benefited from optillm's enhanced reasoning capabilities.

## Available optillm Plugins and Techniques

optillm offers many plugins and optimization techniques. Here are the most useful for code evolution:

### Core Plugins
- **`readurls`**: Automatically fetches web content when URLs are detected in prompts
- **`executecode`**: Runs code and includes output in the response (great for validation)

### Optimization Techniques
- **`moa`** (Mixture of Agents): Generates multiple responses, critiques them, and synthesizes the best
- **`cot_reflection`**: Uses chain-of-thought reasoning with self-reflection
- **`rstar`**: Advanced reasoning technique for complex problems
- **`bon`** (Best of N): Generates N responses and selects the best one
- **`z3_solver`**: Uses Z3 theorem prover for logical reasoning
- **`rto`** (Round Trip Optimization): Optimizes responses through iterative refinement

### Combining Techniques
You can chain multiple techniques using `&`:

```yaml
llm:
  models:
    # Use chain-of-thought + readurls for primary model
    - name: "cot_reflection&readurls-Qwen/Qwen3-1.7B-MLX-bf16"
      weight: 0.7
    # Use MoA + code execution for secondary validation
    - name: "moa&executecode-Qwen/Qwen3-1.7B-MLX-bf16"
      weight: 0.3
```

### Recommended Combinations for Code Evolution
1. **For Documentation-Heavy Tasks**: `cot_reflection&readurls`
2. **For Complex Logic**: `moa&executecode` 
3. **For Mathematical Problems**: `cot_reflection&z3_solver`
4. **For Validation-Critical Code**: `bon&executecode`

## Troubleshooting

1. **optillm not responding**: Ensure it's running on port 8000 with `OPTILLM_API_KEY=optillm`
2. **Model not found**: Make sure optillm's local inference server is working (check optillm logs)
3. **Slow evolution**: MoA generates multiple completions, so it's slower but more accurate

## Further Reading

- [optillm Documentation](https://github.com/codelion/optillm)
- [OpenEvolve Configuration Guide](../../configs/default_config.yaml)
- [Mixture of Agents Paper](https://arxiv.org/abs/2406.04692)