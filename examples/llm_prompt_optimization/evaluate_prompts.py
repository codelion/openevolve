#!/usr/bin/env python3
"""
Unified evaluation script for GEPA benchmark datasets.
Can evaluate baseline or evolved prompts on IFEval, HoVer, and HotpotQA.
"""

import os
import json
import yaml
import time
import argparse
from datetime import datetime
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


# Initialize OpenAI client
def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def load_prompt(dataset_name, prompt_type="baseline"):
    """Load prompt template for a dataset."""
    if prompt_type == "baseline":
        prompt_path = f"{dataset_name}_prompt.txt"
    else:  # evolved
        prompt_path = f"openevolve_output_qwen3_{dataset_name}/best/best_program.txt"

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r") as f:
        return f.read().strip()


def load_dataset_config(dataset_name):
    """Load dataset configuration."""
    config_path = f"{dataset_name}_prompt_dataset.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_ifeval(client, prompt_template, num_samples, model):
    """Evaluate IFEval dataset."""
    print("\nLoading IFEval dataset...")

    # Try test split first, then train
    try:
        dataset = load_dataset("google/IFEval", split="test")
        split_used = "test"
    except:
        dataset = load_dataset("google/IFEval", split="train")
        split_used = "train"

    # Determine samples to process
    if num_samples is None:
        samples_to_process = len(dataset)
        print(f"Using full {split_used} split: {samples_to_process} samples")
        dataset_iter = tqdm(dataset, desc="Evaluating")
    else:
        samples_to_process = min(num_samples, len(dataset))
        print(f"Using {samples_to_process} samples from {split_used} split")
        dataset = load_dataset("google/IFEval", split=split_used, streaming=True)
        dataset_iter = tqdm(
            dataset.take(samples_to_process), total=samples_to_process, desc="Evaluating"
        )

    correct = 0
    total = 0
    empty_responses = 0

    for i, example in enumerate(dataset_iter):
        if num_samples is not None and i >= samples_to_process:
            break
        instruction = example["prompt"]

        try:
            formatted_prompt = prompt_template.format(instruction=instruction)
        except KeyError:
            # Handle prompts with different placeholder names
            formatted_prompt = prompt_template.replace("{instruction}", instruction)

        # Call LLM with retries
        output_text = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": formatted_prompt}],
                    temperature=0.1,
                    max_tokens=4096,
                )

                if response and response.choices and response.choices[0].message:
                    output_text = response.choices[0].message.content
                    if output_text and output_text.strip():
                        break
            except Exception as e:
                if attempt == 2:
                    print(f"\nError after 3 attempts: {e}")
                time.sleep(2)

        if not output_text or not output_text.strip():
            empty_responses += 1
        else:
            # Simple evaluation: response has reasonable length
            if len(output_text.strip()) > 20:
                correct += 1

        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total, empty_responses


def evaluate_hover(client, prompt_template, num_samples, model):
    """Evaluate HoVer dataset."""
    print("\nLoading HoVer dataset...")

    # Try test split first (but it's unlabeled), then validation
    try:
        test_dataset = load_dataset("hover", split="test", trust_remote_code=True)
        # Check if test set has labels
        if test_dataset[0]["label"] != -1:
            dataset = test_dataset
            split_used = "test"
        else:
            # Test set is unlabeled, use validation
            dataset = load_dataset("hover", split="validation", trust_remote_code=True)
            split_used = "validation"
    except:
        dataset = load_dataset("hover", split="validation", trust_remote_code=True)
        split_used = "validation"

    # Determine samples to process
    if num_samples is None:
        samples_to_process = len(dataset)
        print(f"Using full {split_used} split: {samples_to_process} samples")
        dataset_iter = tqdm(dataset, desc="Evaluating")
    else:
        samples_to_process = min(num_samples, len(dataset))
        print(f"Using {samples_to_process} samples from {split_used} split")
        dataset = load_dataset("hover", split=split_used, streaming=True, trust_remote_code=True)
        dataset_iter = tqdm(
            dataset.take(samples_to_process), total=samples_to_process, desc="Evaluating"
        )

    correct = 0
    total = 0
    empty_responses = 0

    for i, example in enumerate(dataset_iter):
        if num_samples is not None and i >= samples_to_process:
            break
        claim = example["claim"]
        label = example["label"]  # Integer: 0=SUPPORTED, 1=NOT_SUPPORTED

        try:
            formatted_prompt = prompt_template.format(claim=claim)
        except KeyError:
            formatted_prompt = prompt_template.replace("{claim}", claim)

        # Call LLM with retries
        output_text = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": formatted_prompt}],
                    temperature=0.1,
                    max_tokens=4096,
                )

                if response and response.choices and response.choices[0].message:
                    output_text = response.choices[0].message.content
                    if output_text and output_text.strip():
                        break
            except Exception as e:
                if attempt == 2:
                    print(f"\nError after 3 attempts: {e}")
                time.sleep(2)

        if not output_text or not output_text.strip():
            empty_responses += 1
        else:
            output_upper = output_text.strip().upper()

            # Parse prediction from output
            if "NOT SUPPORTED" in output_upper or "NOT_SUPPORTED" in output_upper:
                prediction = 1  # NOT_SUPPORTED
            elif "SUPPORTED" in output_upper:
                prediction = 0  # SUPPORTED
            else:
                prediction = -1  # Invalid/unclear response

            # Compare with actual label
            if prediction == label:
                correct += 1

        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total, empty_responses


def evaluate_hotpotqa(client, prompt_template, num_samples, model):
    """Evaluate HotpotQA dataset."""
    print("\nLoading HotpotQA dataset (this may take a moment)...")

    # Try test split first, then validation
    try:
        dataset = load_dataset(
            "hotpotqa/hotpot_qa", "distractor", split="test", trust_remote_code=True
        )
        split_used = "test"
    except:
        dataset = load_dataset(
            "hotpotqa/hotpot_qa", "distractor", split="validation", trust_remote_code=True
        )
        split_used = "validation"

    print(f"Dataset loaded. Using {split_used} split with {len(dataset)} samples")

    # Determine samples to process
    if num_samples is None:
        samples_to_process = len(dataset)
        print(f"Using full dataset: {samples_to_process} samples")
    else:
        samples_to_process = min(num_samples, len(dataset))
        print(f"Using {samples_to_process} samples")

    correct = 0
    total = 0
    empty_responses = 0

    for i in tqdm(range(samples_to_process), desc="Evaluating"):
        example = dataset[i]

        question = example["question"]
        context = example["context"]
        answer = example["answer"].lower().strip()

        # Format context
        context_str = ""
        titles = context["title"]
        sentences = context["sentences"]

        for title, sents in zip(titles, sentences):
            context_str += f"{title}: {' '.join(sents)}\n"

        try:
            formatted_prompt = prompt_template.format(
                context=context_str.strip(), question=question
            )
        except KeyError:
            # Try alternative formatting
            formatted_prompt = prompt_template.replace("{context}", context_str.strip())
            formatted_prompt = formatted_prompt.replace("{question}", question)

        # Call LLM with retries
        output_text = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": formatted_prompt}],
                    temperature=0.1,
                    max_tokens=4096,
                )

                if response and response.choices and response.choices[0].message:
                    output_text = response.choices[0].message.content
                    if output_text and output_text.strip():
                        break
            except Exception as e:
                if attempt == 2:
                    print(f"\nError after 3 attempts: {e}")
                time.sleep(2)

        if not output_text or not output_text.strip():
            empty_responses += 1
        else:
            output_lower = output_text.strip().lower()

            # Check if answer is in output
            if answer in output_lower:
                correct += 1

        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total, empty_responses


def main():
    parser = argparse.ArgumentParser(description="Evaluate prompts on GEPA benchmark datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ifeval", "hover", "hotpotqa", "all"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="baseline",
        choices=["baseline", "evolved"],
        help="Type of prompt to use",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: full dataset)",
    )
    parser.add_argument(
        "--model", type=str, default="qwen/qwen3-8b", help="Model to use for evaluation"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file for results (default: auto-generated)"
    )

    args = parser.parse_args()

    # Initialize client
    client = get_client()

    # Determine which datasets to evaluate
    if args.dataset == "all":
        datasets = ["ifeval", "hover", "hotpotqa"]
    else:
        datasets = [args.dataset]

    # Evaluation functions
    eval_funcs = {"ifeval": evaluate_ifeval, "hover": evaluate_hover, "hotpotqa": evaluate_hotpotqa}

    # Load baseline results for comparison
    baseline_results = {}
    if os.path.exists("baseline_results_50samples.json"):
        with open("baseline_results_50samples.json", "r") as f:
            baseline_data = json.load(f)
            for result in baseline_data.get("results", []):
                baseline_results[result["dataset"]] = result["accuracy"]

    # Store results
    all_results = []

    print(f"\n{'='*60}")
    print(f"PROMPT EVALUATION - {args.prompt_type.upper()}")
    print(f"Model: {args.model}")
    if args.samples:
        print(f"Samples per dataset: {args.samples}")
    else:
        print(f"Samples per dataset: Full dataset")
    print(f"{'='*60}")

    for dataset_name in datasets:
        print(f"\nEvaluating {dataset_name.upper()}...")

        try:
            # Load prompt
            prompt_template = load_prompt(dataset_name, args.prompt_type)
            print(f"Loaded {args.prompt_type} prompt ({len(prompt_template)} chars)")

            # Run evaluation
            start_time = time.time()
            accuracy, correct, total, empty_responses = eval_funcs[dataset_name](
                client, prompt_template, args.samples, args.model
            )
            elapsed_time = time.time() - start_time

            # Get baseline accuracy
            baseline_acc = baseline_results.get(dataset_name)
            if baseline_acc:
                improvement = ((accuracy - baseline_acc) / baseline_acc) * 100
            else:
                improvement = 0

            # Store result
            result = {
                "dataset": dataset_name,
                "prompt_type": args.prompt_type,
                "accuracy": accuracy,
                "baseline_accuracy": baseline_acc,
                "improvement_percent": improvement,
                "correct": correct,
                "total": total,
                "empty_responses": empty_responses,
                "elapsed_time": elapsed_time,
                "timestamp": datetime.now().isoformat(),
            }

            all_results.append(result)

            # Print results
            print(f"\nResults for {dataset_name.upper()}:")
            print(f"  Accuracy: {accuracy:.3f} ({correct}/{total})")
            if baseline_acc:
                print(f"  Baseline: {baseline_acc:.3f}")
                print(f"  Improvement: {improvement:+.1f}%")
            print(f"  Empty responses: {empty_responses}")
            print(f"  Time: {elapsed_time:.1f}s ({elapsed_time/total:.1f}s per sample)")

        except Exception as e:
            print(f"Error evaluating {dataset_name}: {str(e)}")
            all_results.append(
                {
                    "dataset": dataset_name,
                    "prompt_type": args.prompt_type,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            )

    # Save results
    output_path = args.output
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"evaluation_results_{args.prompt_type}_{timestamp}.json"

    final_results = {
        "prompt_type": args.prompt_type,
        "model": args.model,
        "samples_per_dataset": args.samples,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }

    # Calculate aggregate statistics
    valid_results = [r for r in all_results if "error" not in r]
    if valid_results:
        total_correct = sum(r["correct"] for r in valid_results)
        total_samples = sum(r["total"] for r in valid_results)
        aggregate_accuracy = total_correct / total_samples if total_samples > 0 else 0

        final_results["summary"] = {
            "aggregate_accuracy": aggregate_accuracy,
            "total_correct": total_correct,
            "total_samples": total_samples,
            "datasets_evaluated": len(valid_results),
        }

    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")

    for result in all_results:
        if "error" not in result:
            print(f"\n{result['dataset'].upper()}:")
            print(f"  Accuracy: {result['accuracy']:.3f}")
            if result.get("baseline_accuracy"):
                print(f"  vs Baseline: {result['improvement_percent']:+.1f}%")

    if "summary" in final_results:
        print(f"\nAGGREGATE:")
        print(f"  Overall Accuracy: {final_results['summary']['aggregate_accuracy']:.3f}")
        print(f"  Total Samples: {final_results['summary']['total_samples']}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
