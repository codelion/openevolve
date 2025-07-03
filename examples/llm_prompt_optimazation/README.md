# Evolving Better Prompts with OpenEvolve 🧠✨

This example shows how to use **OpenEvolve** to automatically optimize prompts for **Large Language Models (LLMs)**. Whether you're working on classification, summarization, generation, or code tasks, OpenEvolve helps you find high-performing prompts using **evolutionary search**. For this example we'll use syntihetic data for sentiment analysis task, but you can adapt it to your own datasets and tasks.

---

## 🎯 What Is Prompt Optimization?

Prompt engineering is key to getting reliable outputs from LLMs—but finding the right prompt manually can be slow and inconsistent.

OpenEvolve automates this by:

* Generating and evolving prompt variations
* Testing them against your task and metrics
* Selecting the best prompts through generations

You start with a simple prompt and let OpenEvolve evolve it into something smarter and more effective.

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
cd examples/llm_prompt_optimazation
pip install -r requirements.txt
sh run.sh
```

### 2. Add Your models

1. Update your `config.yaml`:

```yaml
llm:
  primary_model: "llm_name"
  api_base: "llm_server_url"
  api_key: "your_api_key_here"
```

2. Update your task-model in `evaluator.py`:

```python
TASK_MODEL_NAME = "task_llm_name"
TASK_MODEL_URL = "task_llm_server_url"
TASK_MODEL_API_KEY = "your_api_key_here"
SAMPLE_SIZE = 25  # Number of samples to use for evaluation
MAX_RETRIES = 3  # Number of retries for LLM calls

```

### 3. Run OpenEvolve

```bash
sh run.sh
```

---

## 🔧 How to Adapt This Template

### 1. Replace the Dataset

Edit `data.json` to match your use case:

```json
[
  {
    "id": 1,
    "input": "Your input here",
    "expected_output": "Target output"
  }
]
```

### 2. Customize the Evaluator

In `evaluator.py`, define how to evaluate a prompt:

* Load your data
* Call the LLM using the prompt
* Measure output quality (accuracy, score, etc.)

### 3. Write Your Initial Prompt

Create a basic starting prompt in `initial_prompt.txt`:

```
# EVOLVE-BLOCK-START
Your task prompt using {input_text} as a placeholder.
# EVOLVE-BLOCK-END
```

This is the part OpenEvolve will improve over time.
Good to add the name of your task in 'initial_prompt.txt' header to help the model understand the context.

---

## ⚙️ Key Config Options (`config.yaml`)

```yaml
llm:
  primary_model: "gpt-4o"           # or your preferred model
  secondary_model: "gpt-3.5"        # optional for diversity
  temperature: 0.9
  max_tokens: 2048

database:
  population_size: 40
  max_iterations: 15
  elite_selection_ratio: 0.25

evaluator:
  timeout: 45
  parallel_evaluations: 3
  use_llm_feedback: true
```

---

## 📈 Example Output

OpenEvolve evolves prompts like this:

**Initial Prompt:**

```
Please analyze the sentiment of the following sentence and provide a sentiment score:

"{input_text}"

Rate the sentiment on a scale from 0.0 to 10.0.

Score:
```

**Evolved Prompt:**

```
Please analyze the sentiment of the following sentence and provide a sentiment score using the following guidelines:
- 0.0-2.9: Strongly negative sentiment (e.g., expresses anger, sadness, or despair)
- 3.0-6.9: Neutral or mixed sentiment (e.g., factual statements, ambiguous content)
- 7.0-10.0: Strongly positive sentiment (e.g., expresses joy, satisfaction, or hope)

"{input_text}"

Rate the sentiment on a scale from 0.0 to 10.0:
- 0.0-2.9: Strongly negative (e.g., "This product is terrible")
- 3.0-6.9: Neutral/mixed (e.g., "The sky is blue today")
- 7.0-10.0: Strongly positive (e.g., "This is amazing!")

Provide only the numeric score (e.g., "8.5") without any additional text:

Score:
```

**Result**: Improved accuracy and output consistency.

---

## 🔍 Where to Use This

OpenEvolve could be addapted on many tasks:

* **Text Classification**: Spam detection, intent recognition
* **Content Generation**: Social media posts, product descriptions
* **Question Answering & Summarization**
* **Code Tasks**: Review, generation, completion
* **Structured Output**: JSON, table filling, data extraction

---

## ✅ Best Practices

* Start with a basic but relevant prompt
* Use good-quality data and clear evaluation metrics
* Run multiple evolutions for better results
* Validate on held-out data before deployment

---

**Ready to discover better prompts?**
Use this template to evolve prompts for any LLM task—automatically.
