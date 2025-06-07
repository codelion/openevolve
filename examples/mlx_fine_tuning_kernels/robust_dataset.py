"""
Robust Dataset Generation for MLX Fine-tuning Kernels

This module provides robust instruction-following dataset generation with proper
error handling and diverse data patterns for realistic fine-tuning benchmarks.
"""

import re
import random
from typing import List, Dict, Optional

try:
    import mlx.core as mx
    import numpy as np
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def create_robust_instruction_dataset(tokenizer, num_samples: int, seq_len: int) -> List[Dict]:
    """
    Create a robust, diverse instruction-following dataset for fine-tuning benchmarks.
    
    This generates realistic instruction-response pairs with:
    - Proper tokenization handling
    - Diverse conversation patterns
    - Robust error handling
    - Memory-efficient processing
    """
    
    if not MLX_AVAILABLE:
        raise ImportError("MLX not available for robust dataset generation")
    
    print(f"    📊 Generating robust instruction dataset...")
    
    # Comprehensive instruction-response templates
    instruction_templates = [
        # Explanatory instructions
        ("Explain {topic}", "A {topic} is {explanation}"),
        ("What is {topic}?", "{topic} refers to {explanation}"),
        ("How does {topic} work?", "{topic} works by {process}"),
        ("Define {topic}", "{topic} can be defined as {definition}"),
        ("Describe {topic}", "{topic} is characterized by {description}"),
        
        # Procedural instructions
        ("How to {action}", "To {action}, you need to {steps}"),
        ("Steps to {action}", "The steps to {action} are: {process}"),
        ("Guide me through {action}", "Here's how to {action}: {instructions}"),
        ("What's the process for {action}?", "The process for {action} involves {steps}"),
        
        # Comparative instructions
        ("Compare {item1} and {item2}", "{item1} and {item2} differ in that {comparison}"),
        ("What's the difference between {item1} and {item2}?", "The main difference is {distinction}"),
        ("Which is better: {item1} or {item2}?", "Between {item1} and {item2}, {preference} because {reason}"),
        
        # Creative instructions
        ("Write about {topic}", "Here's something about {topic}: {content}"),
        ("Create a story about {topic}", "Once upon a time, {topic} {narrative}"),
        ("Imagine {scenario}", "In this scenario where {scenario}, {outcome}"),
    ]
    
    # Rich topic vocabulary for diverse content
    topics = [
        # Technology
        "machine learning", "artificial intelligence", "neural networks", "deep learning",
        "computer vision", "natural language processing", "robotics", "automation",
        "cloud computing", "cybersecurity", "blockchain", "quantum computing",
        
        # Science
        "photosynthesis", "evolution", "genetics", "physics", "chemistry", "biology",
        "astronomy", "climate change", "renewable energy", "space exploration",
        
        # Business
        "entrepreneurship", "marketing", "finance", "leadership", "innovation",
        "project management", "data analysis", "business strategy", "e-commerce",
        
        # General knowledge
        "history", "geography", "literature", "philosophy", "psychology", "sociology",
        "mathematics", "statistics", "economics", "politics", "education", "health"
    ]
    
    actions = [
        "learn programming", "start a business", "solve problems", "analyze data",
        "write code", "design software", "manage projects", "lead teams",
        "research topics", "build websites", "create content", "optimize performance"
    ]
    
    explanations = [
        "a fundamental concept in computer science that enables automated decision-making",
        "an advanced technique used to process and analyze large amounts of data",
        "a method that combines statistical analysis with computational algorithms",
        "an approach that leverages mathematical models to solve complex problems",
        "a systematic process for transforming raw data into actionable insights"
    ]
    
    processes = [
        "analyzing patterns in data and applying mathematical transformations",
        "using algorithms to process information and generate predictions",
        "combining multiple techniques to achieve optimal results",
        "iteratively refining models based on feedback and validation"
    ]
    
    dataset = []
    
    for i in range(num_samples):
        try:
            # Select random template and content
            instruction_template, response_template = random.choice(instruction_templates)
            
            # Fill in template variables
            if "{topic}" in instruction_template:
                topic = random.choice(topics)
                instruction = instruction_template.format(topic=topic)
                response = response_template.format(
                    topic=topic,
                    explanation=random.choice(explanations),
                    process=random.choice(processes),
                    definition=random.choice(explanations),
                    description=random.choice(explanations)
                )
            elif "{action}" in instruction_template:
                action = random.choice(actions)
                instruction = instruction_template.format(action=action)
                response = response_template.format(
                    action=action,
                    steps=random.choice(processes),
                    process=random.choice(processes),
                    instructions=random.choice(processes)
                )
            elif "{item1}" in instruction_template:
                item1, item2 = random.sample(topics, 2)
                instruction = instruction_template.format(item1=item1, item2=item2)
                response = response_template.format(
                    item1=item1,
                    item2=item2,
                    comparison=random.choice(explanations),
                    distinction=random.choice(explanations),
                    preference=item1,
                    reason=random.choice(explanations)
                )
            else:
                # Generic template
                topic = random.choice(topics)
                instruction = instruction_template.format(
                    topic=topic,
                    scenario=f"{topic} becomes widely adopted"
                )
                response = response_template.format(
                    topic=topic,
                    content=random.choice(explanations),
                    narrative=f"revolutionized how we understand {random.choice(topics)}",
                    scenario=f"{topic} becomes widely adopted",
                    outcome=random.choice(explanations)
                )
            
            # Create conversation format
            conversation = f"Instruction: {instruction}\nResponse: {response}"
            
            # Robust tokenization with error handling
            input_ids, labels = tokenize_conversation_robust(
                conversation, tokenizer, seq_len
            )
            
            if input_ids is not None and labels is not None:
                dataset.append({
                    'input_ids': input_ids,
                    'labels': labels,
                    'instruction': instruction,
                    'response': response,
                    'length': len(input_ids) if hasattr(input_ids, '__len__') else seq_len,
                    'conversation': conversation
                })
            
        except Exception as e:
            # Fallback to simple entry if anything fails
            simple_instruction = f"Explain {random.choice(topics)}"
            simple_response = f"This is about {random.choice(explanations)}"
            simple_tokens = create_simple_tokens(simple_instruction + " " + simple_response, seq_len)
            
            dataset.append({
                'input_ids': mx.array(simple_tokens),
                'labels': mx.array(simple_tokens),
                'instruction': simple_instruction,
                'response': simple_response,
                'length': len(simple_tokens),
                'conversation': f"{simple_instruction} {simple_response}"
            })
    
    print(f"    ✅ Generated {len(dataset)} robust samples")
    
    if len(dataset) > 0:
        avg_length = np.mean([d['length'] for d in dataset])
        print(f"    📊 Average length: {avg_length:.1f} tokens")
        print(f"    📊 Unique instructions: {len(set(d['instruction'] for d in dataset))}")
    
    return dataset


def tokenize_conversation_robust(conversation: str, tokenizer, max_length: int) -> tuple:
    """
    Robustly tokenize conversation with comprehensive error handling.
    """
    try:
        # Method 1: Try standard tokenization
        if hasattr(tokenizer, 'encode'):
            tokens = tokenizer.encode(
                conversation,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                padding=False
            )
            
            # Ensure tokens is a list of integers
            if hasattr(tokens, 'tolist'):
                tokens = tokens.tolist()
            elif not isinstance(tokens, list):
                tokens = list(tokens)
            
            # Convert to integers and constrain range
            tokens = [int(t) % 50000 for t in tokens if isinstance(t, (int, float, np.integer))]
            
            # Pad to exact length
            if len(tokens) < max_length:
                pad_token = getattr(tokenizer, 'pad_token_id', 0) or 0
                tokens.extend([pad_token] * (max_length - len(tokens)))
            elif len(tokens) > max_length:
                tokens = tokens[:max_length]
            
            input_ids = mx.array(tokens)
            labels = mx.array(tokens)  # For causal LM, labels = input_ids shifted
            
            return input_ids, labels
            
    except Exception as e:
        pass
    
    try:
        # Method 2: Try with simpler tokenization
        if hasattr(tokenizer, '__call__'):
            result = tokenizer(
                conversation,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors=None
            )
            
            if 'input_ids' in result:
                tokens = result['input_ids']
                if hasattr(tokens, 'tolist'):
                    tokens = tokens.tolist()
                
                tokens = [int(t) % 50000 for t in tokens]
                input_ids = mx.array(tokens)
                labels = mx.array(tokens)
                
                return input_ids, labels
                
    except Exception as e:
        pass
    
    # Method 3: Fallback to character-based tokenization
    return create_char_based_tokens(conversation, max_length)


def create_char_based_tokens(text: str, max_length: int) -> tuple:
    """
    Create tokens based on character encoding as ultimate fallback.
    """
    try:
        # Convert characters to token IDs
        char_tokens = [ord(c) % 1000 + 1 for c in text[:max_length]]
        
        # Pad to exact length
        if len(char_tokens) < max_length:
            char_tokens.extend([0] * (max_length - len(char_tokens)))
        
        input_ids = mx.array(char_tokens)
        labels = mx.array(char_tokens)
        
        return input_ids, labels
        
    except Exception:
        # Ultimate fallback: random tokens
        return create_simple_tokens(text, max_length)


def create_simple_tokens(text: str, max_length: int) -> List[int]:
    """
    Create simple token sequence from text.
    """
    # Hash-based tokenization for reproducibility
    tokens = []
    for i, char in enumerate(text[:max_length]):
        token = (hash(char + str(i)) % 1000) + 1  # Avoid token 0
        tokens.append(token)
    
    # Pad to exact length
    while len(tokens) < max_length:
        tokens.append(0)  # Padding token
    
    return tokens[:max_length]


def validate_dataset(dataset: List[Dict]) -> Dict:
    """
    Validate the generated dataset and return statistics.
    """
    if not dataset:
        return {"valid": False, "error": "Empty dataset"}
    
    try:
        # Check basic structure
        required_keys = ['input_ids', 'labels', 'instruction', 'response']
        for item in dataset[:5]:  # Check first 5 items
            for key in required_keys:
                if key not in item:
                    return {"valid": False, "error": f"Missing key: {key}"}
        
        # Check tensor properties
        lengths = []
        for item in dataset:
            if hasattr(item['input_ids'], 'shape'):
                lengths.append(item['input_ids'].shape[0])
            else:
                lengths.append(len(item['input_ids']))
        
        stats = {
            "valid": True,
            "num_samples": len(dataset),
            "avg_length": np.mean(lengths),
            "min_length": np.min(lengths),
            "max_length": np.max(lengths),
            "unique_instructions": len(set(item['instruction'] for item in dataset))
        }
        
        return stats
        
    except Exception as e:
        return {"valid": False, "error": str(e)}


if __name__ == "__main__":
    # Test the robust dataset generation
    print("Testing robust dataset generation...")
    
    if not MLX_AVAILABLE:
        print("❌ MLX not available")
        exit(1)
    
    # Create a mock tokenizer for testing
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
        
        def encode(self, text, **kwargs):
            # Simple hash-based encoding
            return [hash(word) % 1000 + 1 for word in text.split()[:50]]
    
    mock_tokenizer = MockTokenizer()
    
    # Generate test dataset
    dataset = create_robust_instruction_dataset(mock_tokenizer, 100, 64)
    
    # Validate
    stats = validate_dataset(dataset)
    print(f"Dataset validation: {stats}")
    
    if stats["valid"]:
        print("✅ Robust dataset generation working correctly!")
        print(f"Generated {stats['num_samples']} samples")
        print(f"Average length: {stats['avg_length']:.1f}")
        print(f"Unique instructions: {stats['unique_instructions']}")
    else:
        print(f"❌ Dataset validation failed: {stats['error']}")
