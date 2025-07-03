"""
Evaluator for the prompt optimization task.
"""

import re
import traceback
import json
import os
import time
from openai import OpenAI
from tqdm import tqdm

TASK_MODEL_NAME = "meta-llama-3.1-8b-instruct@q8_0"
TASK_MODEL_URL = "http://localhost:1234/v1"
TASK_MODEL_API_KEY = "your_api_key_here"
SAMPLE_SIZE = 25  # Number of samples to use for evaluation
MAX_RETRIES = 3  # Number of retries for LLM calls


def load_dataset(data_file_path):
    """
    Load the book review dataset from JSON file.
    
    Args:
        data_file_path: Path to the JSON data file
        
    Returns:
        List of review dictionaries with 'text' and 'label' keys
    """
    try:
        with open(data_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert the data structure to match the expected format
        reviews = []
        for review in data.get('book_reviews', []):
            reviews.append({
                'text': review['text'],
                'label': review['sentiment_score']
            })
        
        print(f"Successfully loaded {len(reviews)} book reviews from dataset")
        return reviews
        
    except Exception as e:
        print(f"Error loading dataset from {data_file_path}: {e}")
        traceback.print_exc()
        return []

# Load dataset from JSON file
data_file_path = os.path.join(os.path.dirname(__file__), "data.json")
ds = load_dataset(data_file_path)
        
if not ds:
    raise ValueError("Failed to load dataset or dataset is empty")
  
def evaluate(prompt_path):
    """
    Evaluate the program by run the LLM model on a benchmarck dataset.

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    print('-' * 80)
    print("Starting evaluation...")
    print('-' * 80)
    try:
        # Initialize OpenAI test_model with error handling
        try:
            test_model = OpenAI(
                base_url=TASK_MODEL_URL,
                api_key=TASK_MODEL_API_KEY
            )
            print(f"Initialized OpenAI test_model with model: {TASK_MODEL_NAME}")
        except Exception as e:
            print(f"Error initializing OpenAI test_model: {e}")
            test_model = None

        # Use a subset for faster evaluation during evolution (can be configured)
        eval_sample_size = min(SAMPLE_SIZE, len(ds))  
        ds_sample = ds[:eval_sample_size]   
        print(f"Using {len(ds_sample)} samples from {len(ds)} total reviews for evaluation")
                
        # load the prompt from the file
        with open(prompt_path, "r") as f:
            prompt = f.read()

        # extract the prompt between the markers
        prompt_match = re.search(r"EVOLVE-BLOCK-START(.*)EVOLVE-BLOCK-END", prompt, re.DOTALL)
        if prompt_match:
            prompt = prompt_match.group(1).strip()
        else:
            raise ValueError("No EVOLVE-BLOCK found in the prompt file")
        
        total_score = 0.0
        total_examples = 0
        individual_scores = []

        print(f"Evaluating with prompt:\n{prompt}\n")
        for example in tqdm(ds_sample, desc="Evaluating examples", unit="example"):
            total_examples += 1
            input_text = example["text"]
            expected_score = example["label"]

            # Prepare the message for the LLM
            messages = [
                {"role": "user", "content": prompt.format(input_text=input_text)}
            ]

            # Call the LLM with retry logic
            max_retries = MAX_RETRIES
            for attempt in range(max_retries):
                try:
                    response = test_model.chat.completions.create(
                        model=TASK_MODEL_NAME,
                        messages=messages
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Failed to get response after {max_retries} attempts: {e}")
                        raise e
                    time.sleep(1)  # Brief pause before retry

            output_text = response.choices[0].message.content.strip()
            
            # Extract numerical score from the response
            try:
                # Try to extract a number between 0 and 10
                score_match = re.search(r'(\d+(?:\.\d+)?)', output_text)
                if score_match:
                    predicted_score = float(score_match.group(1))
                    
                    # Ensure score is within valid range (0-10)
                    predicted_score = max(0.0, min(10.0, predicted_score))
                else:
                    predicted_score = 5.0  # Default to neutral
                
                # Calculate accuracy based on how close the prediction is to the expected score
                # Using 1 - (absolute difference / 10), so perfect match = 1.0, worst case = 0.0
                accuracy = 1.0 - (abs(predicted_score - expected_score) / 10.0)
                individual_scores.append(accuracy)
                total_score += accuracy
                
            except Exception as e:
                print(f"Error processing response '{output_text}': {e}")
                individual_scores.append(0.0)  # Score 0 for failed predictions
        # Calculate comprehensive metrics
        average_score = total_score / total_examples if total_examples > 0 else 0.0
        min_score = min(individual_scores) if individual_scores else 0.0
        max_score = max(individual_scores) if individual_scores else 0.0
        
        # Calculate additional metrics
        std_dev = 0.0
        if len(individual_scores) > 1:
            mean = sum(individual_scores) / len(individual_scores)
            variance = sum((x - mean) ** 2 for x in individual_scores) / len(individual_scores)
            std_dev = variance ** 0.5
        
        # Count high-accuracy predictions (>0.8 accuracy)
        high_accuracy_count = sum(1 for score in individual_scores if score > 0.8)
        high_accuracy_rate = high_accuracy_count / len(individual_scores) if individual_scores else 0.0

        print(f"Total examples: {total_examples}")
        print(f"Average accuracy: {average_score:.3f}")
        print(f"Standard deviation: {std_dev:.3f}")
        print(f"Min accuracy: {min_score:.3f}")
        print(f"Max accuracy: {max_score:.3f}")
        print(f"High accuracy rate (>0.8): {high_accuracy_rate:.3f}")
        print('-' * 80)
        return {
            "score": average_score,
            "total_examples": total_examples,
            "individual_scores": individual_scores,
            "min_score": min_score,
            "max_score": max_score,
            "std_dev": std_dev,
            "high_accuracy_rate": high_accuracy_rate
        }

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        traceback.print_exc()
        print('-' * 80)
        return {
            "score": 0.0,
            "total_examples": 0,
            "individual_scores": [],
            "min_score": 0.0,
            "max_score": 0.0,
            "std_dev": 0.0,
            "high_accuracy_rate": 0.0
        }
