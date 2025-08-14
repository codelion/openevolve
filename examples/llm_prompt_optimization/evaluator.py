"""
Evaluator for HuggingFace dataset-based prompt optimization.
"""

import re
import traceback
import yaml
import os
import time
from openai import OpenAI
from tqdm import tqdm
from datasets import load_dataset

# Read config.yaml to get model settings
with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

# Get model settings from config
llm_config = config.get("llm", {})
api_base = llm_config.get("api_base", "http://localhost:1234/v1")

# Handle both single model and model list configurations
models = llm_config.get("models", [])
if models:
    # Use first model from list
    TASK_MODEL_NAME = models[0].get("name", "default-model")
else:
    # Fallback to direct model specification
    TASK_MODEL_NAME = llm_config.get("primary_model", "default-model")

# Get evaluator settings
evaluator_config = config.get("evaluator", {})
MAX_RETRIES = evaluator_config.get("max_retries", 3)

# Get max_tokens from LLM config
MAX_TOKENS = llm_config.get("max_tokens", 16000)
print(f"Using max_tokens: {MAX_TOKENS}")

# Initialize OpenAI client once for all evaluations
test_model = OpenAI(base_url=api_base)
print(f"Initialized OpenAI client with model: {TASK_MODEL_NAME}")

# Determine which dataset to use based on the OPENEVOLVE_PROMPT environment variable
import sys

prompt_file = os.environ.get("OPENEVOLVE_PROMPT")
if not prompt_file:
    # Default to a generic dataset config if not using the wrapper script
    evaluator_dir = os.path.dirname(os.path.abspath(__file__))
    DATASET_CONFIG_PATH = os.path.join(evaluator_dir, "dataset_config.yaml")
    print("Warning: OPENEVOLVE_PROMPT not set. Using default dataset_config.yaml")
else:
    basename = os.path.basename(prompt_file)
    dataset_filename = basename.replace("_prompt.txt", "_prompt_dataset.yaml").replace(
        ".txt", "_dataset.yaml"
    )
    evaluator_dir = os.path.dirname(os.path.abspath(__file__))
    DATASET_CONFIG_PATH = os.path.join(evaluator_dir, dataset_filename)
    print(f"Dataset configuration: {dataset_filename}")


def calculate_prompt_features(prompt):
    """
    Calculate custom features for MAP-Elites binning

    Returns:
        tuple: (prompt_length, reasoning_strategy) - both in range 0-9
    """
    # Feature 1: Prompt length bin (0-9)
    length = len(prompt)
    if length < 100:
        prompt_length = 0  # Minimal
    elif length < 200:
        prompt_length = 1  # Very short
    elif length < 400:
        prompt_length = 2  # Short
    elif length < 600:
        prompt_length = 3  # Medium-short
    elif length < 900:
        prompt_length = 4  # Medium
    elif length < 1200:
        prompt_length = 5  # Medium-long
    elif length < 1600:
        prompt_length = 6  # Long
    elif length < 2000:
        prompt_length = 7  # Very long
    elif length < 2500:
        prompt_length = 8  # Extensive
    else:
        prompt_length = 9  # Very extensive

    # Feature 2: Reasoning strategy (0-9)
    prompt_lower = prompt.lower()

    # Check for few-shot examples
    has_example = (
        "example" in prompt_lower
        or prompt.count("####") >= 4
        or bool(re.search(r"problem:.*?solution:", prompt_lower, re.DOTALL))
    )

    # Check for Chain-of-Thought (CoT) indicators
    has_cot = (
        "step by step" in prompt_lower
        or "step-by-step" in prompt_lower
        or any(phrase in prompt_lower for phrase in ["think through", "reasoning", "explain your"])
        or bool(re.search(r"(first|then|next|finally)", prompt_lower))
    )

    # Assign reasoning strategy bins
    if has_example:
        # Few-shot examples (bins 7-9)
        if has_cot:
            reasoning_strategy = 9  # Few-shot + CoT (most sophisticated)
        elif length > 1500:
            reasoning_strategy = 8  # Extensive few-shot
        else:
            reasoning_strategy = 7  # Basic few-shot
    elif has_cot:
        # Chain-of-thought (bins 4-6)
        if "must" in prompt_lower or "exactly" in prompt_lower:
            reasoning_strategy = 6  # Strict CoT
        elif length > 500:
            reasoning_strategy = 5  # Detailed CoT
        else:
            reasoning_strategy = 4  # Basic CoT
    else:
        # Basic prompts (bins 0-3)
        if length < 100:
            reasoning_strategy = 0  # Minimal
        elif "solve" in prompt_lower or "calculate" in prompt_lower:
            reasoning_strategy = 2  # Direct instruction
        else:
            reasoning_strategy = 1  # Simple prompt

    return prompt_length, reasoning_strategy


def load_prompt_config(prompt_path):
    """Load the prompt from text file and dataset config from matching _dataset.yaml file."""
    # Load prompt from text file
    with open(prompt_path, "r") as f:
        prompt = f.read().strip()

    # Load the configuration (already determined from environment variable)
    if not os.path.exists(DATASET_CONFIG_PATH):
        raise FileNotFoundError(f"Dataset configuration not found: {DATASET_CONFIG_PATH}")

    with open(DATASET_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    return config, prompt


def load_hf_dataset(config):
    """Load HuggingFace dataset based on configuration."""
    dataset_name = config["dataset_name"]
    dataset_config = config.get("dataset_config", None)
    split = config.get("split", "test")
    trust_remote_code = config.get("trust_remote_code", True)  # Default to True for convenience

    print(f"Loading dataset: {dataset_name}")

    # Special handling for HotpotQA - always use non-streaming mode
    if dataset_name == "hotpot_qa" or config.get("is_hotpotqa", False):
        print("Using non-streaming mode for HotpotQA to avoid PyArrow issues")
        streaming = False
    else:
        # For other datasets, use streaming if not specified
        streaming = config.get("streaming", True)

    try:
        # Try to load the specified split
        if dataset_config:
            dataset = load_dataset(
                dataset_name,
                dataset_config,
                split=split,
                trust_remote_code=trust_remote_code,
                streaming=streaming,
            )
        else:
            dataset = load_dataset(
                dataset_name, split=split, trust_remote_code=trust_remote_code, streaming=streaming
            )
    except:
        # Fallback to train split if test is not available
        print(f"Split '{split}' not found, falling back to 'train'")
        if dataset_config:
            dataset = load_dataset(
                dataset_name,
                dataset_config,
                split="train",
                trust_remote_code=trust_remote_code,
                streaming=streaming,
            )
        else:
            dataset = load_dataset(
                dataset_name,
                split="train",
                trust_remote_code=trust_remote_code,
                streaming=streaming,
            )

    # Print dataset info
    if hasattr(dataset, "__len__"):
        print(f"Dataset loaded with {len(dataset)} examples")
    else:
        print(f"Dataset loaded (streaming mode)")

    return dataset


def evaluate_prompt(prompt, dataset, config, num_samples):
    """Evaluate a prompt on a subset of the dataset."""
    input_field = config["input_field"]
    target_field = config["target_field"]

    # Check dataset type
    dataset_name = config.get("dataset_name", "").lower()
    is_emotion = "emotion" in dataset_name
    is_gsm8k = "gsm8k" in dataset_name
    is_hotpotqa = config.get("is_hotpotqa", False)
    is_ifeval = config.get("is_ifeval", False)
    is_hover = config.get("is_hover", False)

    # Sample from dataset - handle both streaming and non-streaming
    if hasattr(dataset, "take"):
        # Streaming dataset
        samples = dataset.take(num_samples)
        sample_iter = tqdm(samples, desc=f"Evaluating {num_samples} samples", total=num_samples)
    else:
        # Non-streaming dataset
        indices = range(min(num_samples, len(dataset)))
        samples = dataset.select(indices)
        sample_iter = tqdm(samples, desc=f"Evaluating {num_samples} samples")

    correct = 0
    total = 0

    for example in sample_iter:
        input_text = example[input_field]
        expected = example[target_field]

        # Prepare the prompt with appropriate formatting
        if is_hotpotqa:
            # Format context from paragraphs
            context_items = example.get("context", {})
            context_text = ""
            if "title" in context_items and "sentences" in context_items:
                # Handle the specific structure of HotpotQA
                for i, (title, sentences) in enumerate(
                    zip(context_items["title"], context_items["sentences"])
                ):
                    context_text += f"Paragraph {i+1} ({title}):\n"
                    context_text += " ".join(sentences) + "\n\n"
            formatted_prompt = prompt.format(context=context_text.strip(), question=input_text)
        elif is_ifeval:
            # IFEval uses 'prompt' field directly
            formatted_prompt = prompt.format(instruction=input_text)
        elif is_hover:
            # HoVer uses claim field
            formatted_prompt = prompt.format(claim=input_text)
        else:
            # Default formatting for other datasets
            formatted_prompt = prompt.format(input_text=input_text)

        # Prepare the message for the LLM
        messages = [{"role": "user", "content": formatted_prompt}]

        # Call the LLM with retry logic
        for attempt in range(MAX_RETRIES):
            try:
                # Use max_tokens from config
                response = test_model.chat.completions.create(
                    model=TASK_MODEL_NAME,
                    messages=messages,
                    temperature=0.1,  # Low temperature for consistent results
                    max_tokens=MAX_TOKENS,
                )
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed to get response after {MAX_RETRIES} attempts: {e}")
                    raise e
                time.sleep(1)

        # Handle potential None response
        if not response:
            print(f"Warning: No response object from LLM")
            total += 1  # Count as incorrect
            continue

        if not response.choices:
            print(f"Warning: No choices in response from LLM")
            total += 1  # Count as incorrect
            continue

        if not response.choices[0].message:
            print(f"Warning: No message in response choice")
            total += 1  # Count as incorrect
            continue

        output_text = response.choices[0].message.content
        if output_text is None:
            print(f"Warning: None content in LLM response")
            print(f"Full response: {response}")
            total += 1  # Count as incorrect
            continue

        output_text = output_text.strip()

        # Extract prediction from output
        try:
            if is_gsm8k:
                # For GSM8K, extract the numeric answer after ####
                # First, extract the expected answer from the ground truth
                expected_answer = expected.split("####")[-1].strip()
                try:
                    expected_number = float(expected_answer.replace(",", ""))
                except:
                    print(f"Warning: Could not parse expected answer: {expected_answer}")
                    total += 1
                    continue

                # Extract prediction from model output
                prediction = None
                if "####" in output_text:
                    predicted_answer = output_text.split("####")[-1].strip()
                    # Extract just the number, removing any extra text like $ signs
                    import re

                    numbers = re.findall(r"-?\$?[\d,]+\.?\d*", predicted_answer)
                    if numbers:
                        try:
                            # Remove $ and , from the number
                            number_str = numbers[0].replace("$", "").replace(",", "")
                            prediction = float(number_str)
                        except:
                            pass

                # If we found a prediction, check if it matches
                if prediction is not None:
                    # Check if answers match (with small tolerance for floats)
                    if abs(prediction - expected_number) < 0.001:
                        correct += 1

                total += 1
                continue  # Skip the general case to avoid double counting

            elif is_hotpotqa:
                # For HotpotQA, do exact match comparison (case-insensitive)
                output_lower = output_text.lower().strip()
                expected_lower = str(expected).lower().strip()

                # Remove common punctuation for better matching
                output_lower = output_lower.rstrip(".,!?;:")
                expected_lower = expected_lower.rstrip(".,!?;:")

                if output_lower == expected_lower:
                    correct += 1
                elif expected_lower in output_lower:
                    # Partial credit if answer is contained in response
                    correct += 1

                total += 1
                continue

            elif is_ifeval:
                # For IFEval, we need more complex evaluation
                # For now, do basic keyword matching
                # Note: Full IFEval requires checking multiple constraints
                output_lower = output_text.lower()

                # Simple heuristic: check if response seems to follow instruction format
                if len(output_text.strip()) > 10:  # Non-trivial response
                    correct += 1  # Simplified - real IFEval needs constraint checking

                total += 1
                continue

            elif is_hover:
                # For HoVer, check if prediction matches SUPPORTED/NOT_SUPPORTED
                output_upper = output_text.upper()
                expected_upper = str(expected).upper()

                # Look for the verdict in the output
                if "SUPPORTED" in output_upper and "NOT" not in output_upper.replace(
                    "NOT SUPPORTED", ""
                ):
                    prediction = "SUPPORTED"
                elif "NOT SUPPORTED" in output_upper or "NOT_SUPPORTED" in output_upper:
                    prediction = "NOT_SUPPORTED"
                else:
                    prediction = None

                if prediction == expected_upper:
                    correct += 1

                total += 1
                continue

            elif is_emotion:
                # For emotion classification (0-5)
                numbers = re.findall(r"\b[0-5]\b", output_text)
                if numbers:
                    prediction = int(numbers[-1])  # Use the last number found
                else:
                    # Try to infer from emotion keywords
                    output_lower = output_text.lower()
                    emotion_map = {
                        "sadness": 0,
                        "sad": 0,
                        "joy": 1,
                        "happy": 1,
                        "happiness": 1,
                        "love": 2,
                        "anger": 3,
                        "angry": 3,
                        "fear": 4,
                        "afraid": 4,
                        "scared": 4,
                        "surprise": 5,
                        "surprised": 5,
                    }
                    prediction = -1
                    for emotion, label in emotion_map.items():
                        if emotion in output_lower:
                            prediction = label
                            break
            else:
                # For sentiment classification (0-1)
                numbers = re.findall(r"\b[01]\b", output_text)
                if numbers:
                    prediction = int(numbers[-1])  # Use the last number found
                else:
                    # Try to infer from keywords
                    output_lower = output_text.lower()
                    if "positive" in output_lower:
                        prediction = 1
                    elif "negative" in output_lower:
                        prediction = 0
                    else:
                        prediction = -1  # Invalid prediction

            if prediction == expected:
                correct += 1

            total += 1

        except Exception as e:
            print(f"Error parsing response '{output_text}': {e}")
            total += 1  # Count as incorrect

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def evaluate_stage1(prompt_path):
    """
    Stage 1 evaluation: Quick evaluation with 10% of samples

    Args:
        prompt_path: Path to the prompt file

    Returns:
        Dictionary with combined_score metric
    """
    print("-" * 80)
    print("Starting Stage 1 evaluation...")
    print("-" * 80)

    try:
        # Load prompt configuration
        config, prompt = load_prompt_config(prompt_path)
        print(f"Loaded prompt configuration")

        # Load dataset
        dataset = load_hf_dataset(config)

        # Get number of samples from config
        num_samples = config.get("max_samples", 50)
        # Fixed to 10 samples for Stage 1 (quick evaluation)
        stage1_samples = 10

        print(f"Stage 1: Evaluating {stage1_samples} samples...")

        # Run evaluation
        accuracy, correct, total = evaluate_prompt(prompt, dataset, config, stage1_samples)

        print(f"Stage 1 accuracy: {accuracy:.3f} ({correct}/{total})")
        print("-" * 80)

        # Calculate custom features
        prompt_length, reasoning_strategy = calculate_prompt_features(prompt)
        print(f"Prompt features - Length bin: {prompt_length}, Reasoning bin: {reasoning_strategy}")

        return {
            "combined_score": accuracy,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_strategy,
        }

    except Exception as e:
        print(f"Stage 1 evaluation failed: {str(e)}")
        traceback.print_exc()
        print("-" * 80)

        # Always return feature dimensions, even on failure
        try:
            # Try to calculate features from the failed prompt
            with open(prompt_path, "r") as f:
                failed_prompt = f.read().strip()
            prompt_length, reasoning_strategy = calculate_prompt_features(failed_prompt)
        except:
            # Fallback values if prompt can't be read
            prompt_length, reasoning_strategy = 0, 0

        return {
            "combined_score": 0.0,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_strategy,
            "error": str(e),
        }


def evaluate_stage2(prompt_path):
    """
    Stage 2 evaluation: Full evaluation with all samples

    Args:
        prompt_path: Path to the prompt file

    Returns:
        Dictionary with combined_score metric
    """
    print("-" * 80)
    print("Starting Stage 2 evaluation...")
    print("-" * 80)

    try:
        # Load prompt configuration
        config, prompt = load_prompt_config(prompt_path)
        print(f"Loaded prompt configuration")

        # Load dataset
        dataset = load_hf_dataset(config)

        # Get number of samples from config
        num_samples = config.get("max_samples", 50)
        # Fixed to 40 samples for Stage 2 (comprehensive evaluation)
        stage2_samples = 40

        print(f"Stage 2: Evaluating {stage2_samples} samples...")

        # Run evaluation
        accuracy, correct, total = evaluate_prompt(prompt, dataset, config, stage2_samples)

        print(f"Stage 2 accuracy: {accuracy:.3f} ({correct}/{total})")
        print("-" * 80)

        # Calculate custom features
        prompt_length, reasoning_strategy = calculate_prompt_features(prompt)
        print(f"Prompt features - Length bin: {prompt_length}, Reasoning bin: {reasoning_strategy}")

        return {
            "combined_score": accuracy,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_strategy,
        }

    except Exception as e:
        print(f"Stage 2 evaluation failed: {str(e)}")
        traceback.print_exc()
        print("-" * 80)

        # Always return feature dimensions, even on failure
        try:
            # Try to calculate features from the failed prompt
            with open(prompt_path, "r") as f:
                failed_prompt = f.read().strip()
            prompt_length, reasoning_strategy = calculate_prompt_features(failed_prompt)
        except:
            # Fallback values if prompt can't be read
            prompt_length, reasoning_strategy = 0, 0

        return {
            "combined_score": 0.0,
            "prompt_length": prompt_length,
            "reasoning_strategy": reasoning_strategy,
            "error": str(e),
        }


def evaluate(prompt_path):
    """
    Main evaluation function - for backwards compatibility
    Calls evaluate_stage2 for full evaluation

    Args:
        prompt_path: Path to the prompt file

    Returns:
        Dictionary with combined_score metric
    """
    return evaluate_stage2(prompt_path)
