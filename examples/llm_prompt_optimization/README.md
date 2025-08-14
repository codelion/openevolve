# LLM Prompt Optimization with OpenEvolve 🚀

This example demonstrates how to use OpenEvolve to automatically optimize prompts for Large Language Models across various benchmark datasets. The system uses evolutionary search to discover high-performing prompts, achieving significant improvements across multiple tasks.

## 📊 Latest Performance Results (GEPA Benchmarks)

OpenEvolve successfully improved prompt performance across three challenging GEPA benchmarks:

| Dataset | Baseline Accuracy | Evolved Accuracy | Improvement | Samples |
|---------|------------------|------------------|-------------|---------|
| **IFEval** | 95.01% | 97.41% | **+2.40%** ✅ | 541 |
| **HoVer** | 43.83% | 42.90% | -0.93% | 4,000 |
| **HotpotQA** | 77.93% | 88.62% | **+10.69%** ✅ | 7,405 |
| **Overall** | 67.29% | 73.71% | **+6.42%** ✅ | 11,946 |

### Key Achievements:
- **767 more correct answers** across all datasets
- **38% fewer empty responses** with evolved prompts
- **Near-perfect performance** on instruction following (IFEval: 97.41%)
- **Major improvement** in multi-hop reasoning (HotpotQA: 88.62%)

## 🎯 Overview

OpenEvolve automatically:
- Evolves prompts through multiple generations using LLMs
- Uses cascading evaluation for efficient testing
- Employs MAP-Elites algorithm to maintain diversity
- Incorporates LLM feedback for qualitative assessment
- Supports various datasets from HuggingFace

## 📊 All Supported Datasets

### GEPA Benchmarks (Latest Focus)

#### IFEval (Instruction Following Eval)
- **Task**: Follow complex, multi-constraint instructions
- **Size**: 541 samples (train split)
- **Metric**: Binary success on instruction adherence
- **Results**: 95.01% → 97.41% (+2.40%)
- **Config**: `ifeval_prompt_dataset.yaml`

#### HoVer (Claim Verification)
- **Task**: Verify claims as SUPPORTED or NOT_SUPPORTED
- **Size**: 4,000 samples (validation split)
- **Metric**: Binary classification accuracy
- **Results**: 43.83% → 42.90% (-0.93%)
- **Config**: `hover_prompt_dataset.yaml`
- **Note**: Uses integer labels (0=SUPPORTED, 1=NOT_SUPPORTED)

#### HotpotQA (Multi-hop Question Answering)
- **Task**: Answer questions requiring reasoning over multiple paragraphs
- **Size**: 7,405 samples (validation split)
- **Metric**: Exact match with answer
- **Results**: 77.93% → 88.62% (+10.69%)
- **Config**: `hotpotqa_prompt_dataset.yaml`

### Additional Datasets (Earlier Experiments)

#### Emotion Classification
- **Task**: Classify emotions in text (6 classes)
- **Dataset**: `dair-ai/emotion`
- **Config**: `emotion_prompt_dataset.yaml`
- **Benchmark**: Compared against DSPy results

#### GSM8K (Grade School Math)
- **Task**: Solve grade school math word problems
- **Dataset**: `gsm8k`
- **Config**: `gsm8k_prompt_dataset.yaml`
- **Benchmark**: DSPy achieves 97.1%

#### IMDB Sentiment Analysis
- **Task**: Binary sentiment classification
- **Dataset**: `stanfordnlp/imdb`
- **Config**: `initial_prompt_dataset.yaml`
- **Example Evolution**: 72% → 94% accuracy

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd examples/llm_prompt_optimization
pip install -r requirements.txt
```

### 2. Set Your API Key

```bash
export OPENAI_API_KEY="your_openrouter_api_key"
```

Note: Despite the variable name, this uses OpenRouter API. Get your key at https://openrouter.ai/

### 3. Evaluate Prompts

Use the unified evaluation script to test baseline or evolved prompts:

```bash
# Evaluate baseline prompts on a single dataset
python evaluate_prompts.py --dataset ifeval --prompt-type baseline --samples 100

# Evaluate evolved prompts on a single dataset
python evaluate_prompts.py --dataset hover --prompt-type evolved --samples 100

# Evaluate all GEPA datasets with evolved prompts (full dataset)
python evaluate_prompts.py --dataset all --prompt-type evolved

# Specify output file
python evaluate_prompts.py --dataset all --prompt-type evolved --output results.json
```

### 4. Run Evolution

To evolve prompts from scratch:

```bash
# For GEPA benchmarks
python ../../openevolve-run.py ifeval_prompt.txt evaluator.py \
  --config config_qwen3_evolution.yaml \
  --iterations 50

# For other datasets (using wrapper script)
./run_evolution.sh emotion_prompt.txt --iterations 50
./run_evolution.sh gsm8k_prompt.txt --iterations 100
```

## ⚙️ Configuration Files

### Evolution Configurations

#### GEPA Benchmarks (`config_qwen3_evolution.yaml`)
```yaml
llm:
  models:
    - name: "qwen/qwen3-8b"
      weight: 1.0
  temperature: 0.7
  max_tokens: 4096

evaluator:
  cascade_evaluation: true
  cascade_thresholds: [0.9]  # 2-stage evaluation
  timeout: 1800  # 30 minutes
  use_llm_feedback: true
  llm_feedback_weight: 0.3

database:
  n_islands: 4  # Island-based evolution
  migration_interval: 10
```

#### General Configuration (`config.yaml`)
```yaml
llm:
  api_base: "https://openrouter.ai/api/v1"
  models:
    - name: "google/gemini-2.5-flash"
      weight: 1.0
```

### Dataset Configurations

Each dataset has its own configuration file following the pattern `*_prompt_dataset.yaml`:

```yaml
# Example: ifeval_prompt_dataset.yaml
dataset_name: "google/IFEval"
input_field: "prompt"
target_field: "instruction_id_list"
split: "train"
is_ifeval: true  # Special handling flag
```

## 🧬 Evolution Process

### How It Works

1. **Initial Population**: Start with baseline prompt
2. **Variation**: LLM generates prompt mutations
3. **Evaluation**: Test on dataset samples (10 for Stage 1, 40 for Stage 2)
4. **Selection**: Keep best performers based on combined score
5. **Island Evolution**: 4 isolated populations with periodic migration
6. **Iteration**: Repeat for specified generations (typically 50-100)

### Cascade Evaluation

- **Stage 1**: Quick test on 10 samples (must achieve 90% to proceed)
- **Stage 2**: Comprehensive test on 40 samples
- **Combined Score**: 70% task accuracy + 30% LLM feedback

### LLM Feedback Metrics

Evolved prompts are evaluated on:
- **Clarity**: Unambiguous instructions
- **Specificity**: Appropriate detail level
- **Robustness**: Edge case handling
- **Format Specification**: Clear output requirements

## 📁 Complete File Structure

```
llm_prompt_optimization/
├── evaluate_prompts.py          # Unified evaluation script
├── evaluator.py                 # OpenEvolve evaluator
├── run_evolution.sh             # Wrapper script for evolution
│
├── Configuration Files
│   ├── config.yaml              # General LLM config
│   ├── config_qwen3_evolution.yaml  # GEPA evolution config
│   └── config_qwen3_baseline.yaml   # GEPA baseline config
│
├── Dataset Configurations & Prompts
│   ├── ifeval_prompt.txt & ifeval_prompt_dataset.yaml
│   ├── hover_prompt.txt & hover_prompt_dataset.yaml
│   ├── hotpotqa_prompt.txt & hotpotqa_prompt_dataset.yaml
│   ├── emotion_prompt.txt & emotion_prompt_dataset.yaml
│   ├── gsm8k_prompt.txt & gsm8k_prompt_dataset.yaml
│   └── initial_prompt.txt & initial_prompt_dataset.yaml
│
├── Evolution Templates
│   └── templates/
│       ├── full_rewrite_user.txt
│       ├── evaluation.txt
│       └── evaluator_system_message.txt
│
├── Results
│   ├── evaluation_results_baseline_20250809_070942.json
│   ├── evaluation_results_evolved_20250809_103002.json
│   └── openevolve_output_qwen3_*/
│       └── best/
│           └── best_program.txt     # Evolved prompt
│
└── requirements.txt
```

## 🔍 Example Evolved Prompts

### IFEval (97.41% accuracy)
```
Follow the instruction below precisely. Structure your response into two 
distinct parts: 1) a step-by-step reasoning process that explicitly 
identifies the task, constraints, and required output format, and 2) the 
final answer in the exact format specified...
```

### HotpotQA (88.62% accuracy)
```
Answer the following question using the provided context. The answer must 
integrate information from multiple paragraphs and follow these steps:
1. Paragraph Analysis: Extract key details from each relevant paragraph...
2. Synthesis: Combine these details into a single, coherent response...
3. Citation: Attribute all assertions to their source paragraphs...
```

### IMDB Sentiment (Example Evolution)
Starting prompt:
```
Analyze the sentiment: "{input_text}"
```

Evolved prompt after 100 iterations:
```
Analyze the sentiment of the following text. Determine if the overall 
emotional tone is positive or negative.

Text: "{input_text}"

Response: Provide only a single digit - either 1 for positive sentiment 
or 0 for negative sentiment. Do not include any explanation or additional text.
```
Accuracy improvement: 72% → 94%

## 🐛 Troubleshooting

### HoVer Dataset Issues
- **Problem**: Test split has no labels (all -1)
- **Solution**: Use validation split (configured automatically)
- **Labels**: Integer format (0=SUPPORTED, 1=NOT_SUPPORTED)

### Empty Responses
- **Cause**: Complex evolved prompts exceeding token limits
- **Solution**: Increase max_tokens in evaluation or simplify prompts

### Slow Evaluation
- **IFEval**: ~1 minute per 100 samples
- **HoVer**: ~30 minutes for full dataset
- **HotpotQA**: ~45 minutes for full dataset
- **Tip**: Use --samples flag for faster testing

### Dataset Not Found
- Check the exact dataset name and source
- Some datasets require acceptance of terms
- Use `trust_remote_code=True` for certain datasets

## 🚀 Advanced Usage

### Custom Datasets

To add a new dataset:

1. Create initial prompt: `mydataset_prompt.txt`
2. Create configuration: `mydataset_prompt_dataset.yaml`
3. Run evolution: 
   ```bash
   ./run_evolution.sh mydataset_prompt.txt --iterations 50
   # or directly:
   python ../../openevolve-run.py mydataset_prompt.txt evaluator.py --config config.yaml
   ```

### Batch Evaluation

Evaluate multiple configurations:

```bash
# Create a script to run multiple evaluations
for dataset in ifeval hover hotpotqa; do
    python evaluate_prompts.py --dataset $dataset --prompt-type evolved
done
```

### Resume Evolution

Continue from a checkpoint:

```bash
python ../../openevolve-run.py prompt.txt evaluator.py \
  --config config_qwen3_evolution.yaml \
  --checkpoint openevolve_output_qwen3_ifeval/checkpoints/checkpoint_30 \
  --iterations 20
```

### Custom Templates

The `templates/` directory contains customizable templates for prompt evolution:
- `full_rewrite_user.txt`: Instructions for prompt rewriting
- `evaluation.txt`: LLM feedback template
- `evaluator_system_message.txt`: System message for evaluation

## 📈 Tips for Best Results

1. **Start Simple**: Begin with clear, working baseline prompts
2. **Sufficient Samples**: Use at least 40 samples for Stage 2 evaluation
3. **Monitor Progress**: Check `openevolve_output_*/logs/` for progress
4. **Multiple Runs**: Evolution has randomness; try multiple runs
5. **Token Limits**: Ensure max_tokens accommodates prompt + response
6. **Dataset Variety**: Test on multiple datasets to ensure generalization

## 📚 References

- [OpenEvolve Documentation](../../README.md)
- [IFEval Paper](https://arxiv.org/abs/2311.07911)
- [HoVer Dataset](https://hover-nlp.github.io/)
- [HotpotQA Paper](https://arxiv.org/abs/1809.09600)
- [GSM8K Dataset](https://github.com/openai/grade-school-math)
- [DSPy Framework](https://github.com/stanfordnlp/dspy)
- [OpenRouter API](https://openrouter.ai/docs)

Happy prompt evolving! 🧬✨