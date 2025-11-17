# Multi-API Configuration Guide

This guide explains how to configure OpenEvolve's hierarchical evolution system to use multiple LLM APIs simultaneously, enabling cost-effective tiered reasoning with different models for different abstraction layers.

## Overview

OpenEvolve's hierarchical evolution system supports using different LLM providers for different reasoning tiers:

- **Tier 0 (L1 Code Details)**: Fast, inexpensive models for high-frequency local optimization
- **Tier 1 (L2 Implementation Patterns)**: Standard models for algorithmic patterns
- **Tier 2 (L3 Architectural Components)**: Stronger models for architectural innovations
- **Tier 3 (L4/L5 Strategic Pivots)**: Advanced reasoning models for paradigm shifts

This approach optimizes both **cost** and **quality** by using the right model for each task.

## Recommended Configuration

### KIMI K2 + GLM-4.6 (Optimized for Chinese/Asian Markets)

**Cost-Performance Analysis:**

| Tier | Layer | Model | Use Case | Relative Cost |
|------|-------|-------|----------|---------------|
| 0 | L1 | GLM-4-Flash | Code tweaks, parameter tuning | 1x (baseline) |
| 1 | L2 | GLM-4-Air | Algorithm patterns, data structures | 3x |
| 2 | L3 | GLM-4-Plus | Architectural components | 10x |
| 3 | L4/L5 | KIMI K2 (moonshot-v1-auto) | Strategic pivots, paradigm shifts | 30x |

**Why This Works:**
- L1 iterations are **10-20x more frequent** than L4, so using a cheap model here saves significant cost
- L4/L5 evolutions are **rare but critical**, justifying the use of advanced reasoning models
- Total cost is ~3-5x lower than using premium models for everything

### Alternative: OpenAI O3 + GPT-4 (Global Markets)

| Tier | Layer | Model | Use Case | Relative Cost |
|------|-------|-------|----------|---------------|
| 0 | L1 | GPT-4o-mini | Code tweaks | 1x |
| 1 | L2 | GPT-4o | Algorithm patterns | 5x |
| 2 | L3 | GPT-4o | Architectural components | 5x |
| 3 | L4/L5 | O3-mini | Strategic pivots | 25x |

## Setup Instructions

### 1. Get API Keys

#### KIMI K2 (Moonshot AI)
1. Visit: https://platform.moonshot.cn/
2. Sign up and verify account
3. Navigate to API Keys section
4. Create a new API key
5. Note: KIMI supports OpenAI-compatible API

#### GLM-4.6 (Zhipu AI)
1. Visit: https://open.bigmodel.cn/
2. Sign up and verify account
3. Navigate to API management
4. Create API credentials
5. Note: GLM uses OpenAI-compatible API format

#### OpenAI (Alternative)
1. Visit: https://platform.openai.com/
2. Sign up or log in
3. Navigate to API keys
4. Create a new secret key

### 2. Set Environment Variables

Create a `.env` file in your project root or export these variables:

```bash
# KIMI K2 API
export KIMI_API_KEY="your-kimi-api-key-here"

# GLM-4.6 API
export GLM_API_KEY="your-glm-api-key-here"

# OpenAI API (if using as alternative)
export OPENAI_API_KEY="your-openai-api-key-here"
```

**Security Note:** Never commit `.env` files to version control. Add `.env` to your `.gitignore`.

### 3. Configure Your Example

OpenEvolve automatically reads API keys from environment variables:
- `KIMI_API_KEY` → Used for models with `api_base: "https://api.moonshot.cn/v1"`
- `GLM_API_KEY` → Used for models with `api_base: "https://open.bigmodel.cn/api/paas/v4"`
- `OPENAI_API_KEY` → Used for models with `api_base: "https://api.openai.com/v1"`

**Important:** The config system uses these environment variable names by default. If you need custom names, you can override by setting `api_key` directly in the model configuration (not recommended for security).

Use the provided multi-API configuration files:

```bash
# For polygon decomposition
cd examples/polygon_decomposition
python run_hierarchical.py --config config_multi_api.yaml --iterations 300

# For circle packing
cd examples/circle_packing_hierarchical
python run_hierarchical.py --config config_multi_api.yaml --iterations 300
```

## Configuration Details

### Tier 0: GLM-4-Flash (L1 Code Details)

```yaml
tier0_models:
  - name: "glm-4-flash"
    api_base: "https://open.bigmodel.cn/api/paas/v4"
    weight: 1.0
    temperature: 0.8  # Higher for exploration
    max_tokens: 2048
    timeout: 60
```

**Characteristics:**
- Fast response times (~1-2s)
- Low cost per token
- High frequency usage (most iterations)
- Good for: Parameter tweaking, minor code changes

### Tier 1: GLM-4-Air (L2 Implementation Patterns)

```yaml
tier1_models:
  - name: "glm-4-air"
    api_base: "https://open.bigmodel.cn/api/paas/v4"
    weight: 1.0
    temperature: 0.7
    max_tokens: 4096
    timeout: 90
```

**Characteristics:**
- Balanced speed and capability
- Medium cost
- Moderate frequency (triggered every 5-15 iterations)
- Good for: Algorithm patterns, data structure changes

### Tier 2: GLM-4-Plus (L3 Architectural Components)

```yaml
tier2_models:
  - name: "glm-4-plus"
    api_base: "https://open.bigmodel.cn/api/paas/v4"
    weight: 1.0
    temperature: 0.6
    max_tokens: 8192
    timeout: 120
```

**Characteristics:**
- Stronger reasoning capabilities
- Higher cost
- Low frequency (triggered every 15-50 iterations)
- Good for: Architectural redesigns, component additions

### Tier 3: KIMI K2 (L4/L5 Strategic Pivots)

```yaml
tier3_models:
  - name: "moonshot-v1-auto"  # KIMI K2
    api_base: "https://api.moonshot.cn/v1"
    weight: 1.0
    temperature: 0.5  # Lower for focused reasoning
    max_tokens: 16384
    timeout: 300
    reasoning_effort: "high"  # Deep reasoning mode
```

**Characteristics:**
- Advanced reasoning (CoT-style thinking)
- Highest cost but rare usage
- Very low frequency (triggered every 50-200 iterations)
- Good for: Paradigm shifts, fundamental approach changes

## Model Selection Guide

### KIMI K2 Models

| Model Name | Description | Best For |
|------------|-------------|----------|
| `moonshot-v1-auto` | Automatic reasoning depth | Most use cases (recommended) |
| `moonshot-v1-8k` | 8K context window | Shorter contexts |
| `moonshot-v1-32k` | 32K context window | Standard use |
| `moonshot-v1-128k` | 128K context window | Large context |

**API Endpoint:** `https://api.moonshot.cn/v1`

### GLM-4.6 Models

| Model Name | Description | Best For |
|------------|-------------|----------|
| `glm-4-flash` | Fastest, cheapest | High-frequency tasks (Tier 0) |
| `glm-4-air` | Balanced performance | Standard tasks (Tier 1) |
| `glm-4-plus` | Strongest reasoning | Complex tasks (Tier 2) |
| `glm-4-0520` | Specific version | Version pinning |

**API Endpoint:** `https://open.bigmodel.cn/api/paas/v4`

### OpenAI Models (Alternative)

| Model Name | Description | Best For |
|------------|-------------|----------|
| `gpt-4o-mini` | Fast, cheap | Tier 0/1 |
| `gpt-4o` | Standard GPT-4 | Tier 1/2 |
| `o3-mini` | Reasoning model | Tier 3 |
| `o1` | Full reasoning | Tier 3 (expensive) |

**API Endpoint:** `https://api.openai.com/v1`

## Cost Optimization Strategies

### 1. Adjust Transition Triggers

Control how quickly evolution escalates to higher (more expensive) tiers:

```yaml
hierarchical:
  l2_plateau_iterations: 5   # Default: Try L2 after 5 L1 plateaus
  l3_plateau_iterations: 15  # Default: Try L3 after 15 L2 plateaus
  l4_plateau_iterations: 50  # Default: Try L4 after 50 L3 plateaus
```

**Cost Impact:**
- **Conservative** (5, 15, 50): Balanced cost/performance ✓
- **Aggressive** (3, 10, 30): Higher cost, faster breakthroughs
- **Patient** (10, 30, 100): Lower cost, slower progress

### 2. Use Token Limits

Limit token usage per tier:

```yaml
tier0_models:
  - max_tokens: 2048   # Tier 0: Small responses
tier1_models:
  - max_tokens: 4096   # Tier 1: Medium responses
tier2_models:
  - max_tokens: 8192   # Tier 2: Large responses
tier3_models:
  - max_tokens: 16384  # Tier 3: Very large (reasoning needs space)
```

### 3. Adjust Timeouts

Prevent runaway costs from hung requests:

```yaml
tier0_models:
  - timeout: 60    # 1 minute
tier3_models:
  - timeout: 300   # 5 minutes (reasoning takes time)
```

### 4. Enable Cascade Evaluation

Stop evaluating programs early if they fail basic tests:

```yaml
evaluator:
  cascade_evaluation: true
  cascade_thresholds: [0.3, 0.6]  # Must pass 30%, then 60% to continue
```

This prevents wasting evaluation time on clearly bad programs.

## Monitoring Costs

### Tracking API Usage

Add logging to monitor which models are being used:

```yaml
database:
  log_prompts: true  # Log all prompts (check tokens used)

log_level: INFO      # See model selection messages
```

### Expected Cost Breakdown

For a 300-iteration run with KIMI K2 + GLM-4.6:

```
L1 iterations: ~250 × GLM-4-Flash   = ~$2-5
L2 iterations: ~40  × GLM-4-Air     = ~$3-6
L3 iterations: ~8   × GLM-4-Plus    = ~$4-8
L4 iterations: ~2   × KIMI K2       = ~$3-6
                                Total: ~$12-25
```

Compare to using KIMI K2 for everything: **~$150-300** (10-15x more expensive!)

## Troubleshooting

### API Key Not Found

**Error:**
```
openai.APIConnectionError: Connection error
```

**Solution:**
```bash
# Verify environment variables are set
echo $KIMI_API_KEY
echo $GLM_API_KEY

# Re-export if needed
export KIMI_API_KEY="your-key"
export GLM_API_KEY="your-key"
```

### Wrong API Endpoint

**Error:**
```
404 Not Found
```

**Solution:**
Check your `api_base` matches the provider:
- KIMI: `https://api.moonshot.cn/v1`
- GLM: `https://open.bigmodel.cn/api/paas/v4`
- OpenAI: `https://api.openai.com/v1`

### Rate Limiting

**Error:**
```
429 Too Many Requests
```

**Solution:**
1. Add retry configuration:
```yaml
llm:
  retries: 5
  retry_delay: 10  # seconds
```

2. Or reduce parallelism:
```yaml
max_iterations: 300  # Reduce from higher value
```

### Timeout Issues

**Error:**
```
asyncio.TimeoutError
```

**Solution:**
Increase timeout for reasoning models:
```yaml
tier3_models:
  - timeout: 600  # 10 minutes for deep reasoning
```

### Model Not Available

**Error:**
```
Invalid model: moonshot-v1-auto
```

**Solution:**
1. Check your API subscription includes the model
2. Verify model name spelling
3. Try alternative model:
```yaml
tier3_models:
  - name: "moonshot-v1-32k"  # Alternative
```

## Advanced Configurations

### Using 3 APIs

Mix KIMI K2, GLM-4.6, and OpenAI:

```yaml
tier0_models:
  - name: "glm-4-flash"
    api_base: "https://open.bigmodel.cn/api/paas/v4"

tier1_models:
  - name: "gpt-4o-mini"
    api_base: "https://api.openai.com/v1"

tier2_models:
  - name: "gpt-4o"
    api_base: "https://api.openai.com/v1"

tier3_models:
  - name: "moonshot-v1-auto"
    api_base: "https://api.moonshot.cn/v1"
```

### Ensemble Within Tier

Use multiple models in a single tier:

```yaml
tier3_models:
  - name: "moonshot-v1-auto"
    api_base: "https://api.moonshot.cn/v1"
    weight: 0.7
  - name: "o3-mini"
    api_base: "https://api.openai.com/v1"
    weight: 0.3
```

This creates an ensemble where 70% of L4/L5 mutations use KIMI K2 and 30% use O3-mini.

### Region-Specific Endpoints

KIMI and GLM may have regional endpoints:

```yaml
tier0_models:
  - name: "glm-4-flash"
    api_base: "https://cn.api.bigmodel.cn/api/paas/v4"  # China region
```

Check provider documentation for available regions.

## Best Practices

### 1. Start with Default Configuration

Use the provided `config_multi_api.yaml` as-is for your first run:

```bash
python run_hierarchical.py --config config_multi_api.yaml --iterations 50
```

Monitor logs to verify models are being selected correctly.

### 2. Tune Transition Triggers

After a test run, adjust based on plateau behavior:

- If L1 plateaus quickly (< 5 iterations): **Decrease** `l2_plateau_iterations`
- If L1 keeps improving: **Increase** `l2_plateau_iterations`

### 3. Monitor Layer Statistics

Check which layers are being activated:

```
📊 Hierarchical Evolution Statistics:
  Layer Status:
    code_details: best=5234.50, attempts=250, success_rate=35%
    implementation_patterns: best=6120.30, attempts=40, success_rate=42%
    architectural_components: best=7890.10, attempts=8, success_rate=50%
```

If `attempts` for Tier 3 is very high (> 20 in 300 iterations), you may be triggering L4 too frequently → increase `l4_plateau_iterations`.

### 4. Track Costs

After each run, estimate costs:

```python
# Calculate approximate costs
tier0_tokens = 250 * 2048  # iterations × avg tokens
tier3_tokens = 2 * 16384   # iterations × avg tokens

tier0_cost = tier0_tokens * 0.0001 / 1000  # $0.0001 per 1K tokens (example)
tier3_cost = tier3_tokens * 0.001 / 1000   # $0.001 per 1K tokens (example)
```

### 5. Use Version Pinning

For reproducibility, pin model versions:

```yaml
tier3_models:
  - name: "moonshot-v1-20250101"  # Specific version
```

### 6. Fallback Configuration

Include fallback models in case primary model is unavailable:

```yaml
tier3_models:
  - name: "moonshot-v1-auto"
    api_base: "https://api.moonshot.cn/v1"
    weight: 1.0
  - name: "gpt-4o"  # Fallback
    api_base: "https://api.openai.com/v1"
    weight: 0.0  # Only used if primary fails
```

## Example: Complete Multi-API Workflow

```bash
# 1. Set up environment
export KIMI_API_KEY="sk-..."
export GLM_API_KEY="glm-..."

# 2. Navigate to example
cd examples/polygon_decomposition

# 3. Run with multi-API config
python run_hierarchical.py \
  --config config_multi_api.yaml \
  --iterations 300

# 4. Monitor progress
# Watch for "Using X ensemble in Y phase" messages
# Check layer statistics every 50 iterations

# 5. Analyze results
python -c "
import json
with open('openevolve_output/checkpoints/checkpoint_300/layer_transitions.json') as f:
    data = json.load(f)
    print(f'Total L4 transitions: {sum(1 for t in data[\"layer_transitions\"] if \"L4\" in t.get(\"layer\", \"\"))}')
"
```

## References

- KIMI K2 Documentation: https://platform.moonshot.cn/docs
- GLM-4.6 Documentation: https://open.bigmodel.cn/dev/howuse/model
- OpenEvolve Hierarchical System: `/docs/HIERARCHICAL_EVOLUTION.md`
- Configuration Reference: `/docs/CONFIGURATION.md`

## Support

For issues with:
- **KIMI K2 API**: Contact Moonshot AI support
- **GLM-4.6 API**: Contact Zhipu AI support
- **OpenEvolve Configuration**: Open an issue on GitHub

## License

This configuration guide is part of OpenEvolve (MIT License).
