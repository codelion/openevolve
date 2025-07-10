"""
Evaluator for web scraper evolution.

This evaluator tests the scraper against real documentation pages,
providing feedback on accuracy and robustness. It includes URLs
that will be fetched by optillm's readurls plugin during evolution.
"""

import sys
import os
import traceback
from typing import Dict, List, Any

# Add the program directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



def evaluate(program_path: str) -> Dict:
    """
    Evaluate the web scraper program.
    
    Args:
        program_path: Path to the program to evaluate
        
    Returns:
        Dictionary with metrics and artifacts for OpenEvolve compatibility
    """
    try:
        # Import the program
        sys.path.insert(0, os.path.dirname(program_path))
        program_name = os.path.basename(program_path).replace('.py', '')
        program = __import__(program_name)
        
        # Test data: HTML content from various documentation sources
        test_cases = get_test_cases()
        
        # Evaluate each test case
        metrics = {
            'accuracy': 0.0,
            'completeness': 0.0,
            'robustness': 0.0,
            'parsing_errors': 0.0,
            'total_score': 0.0
        }
        
        artifacts = {}
        
        total_correct = 0
        total_expected = 0
        parsing_errors = 0
        
        for i, test_case in enumerate(test_cases):
            try:
                # Run the scraper
                docs = program.scrape_api_docs(test_case['html'])
                
                # Evaluate accuracy
                correct, expected = evaluate_extraction(docs, test_case['expected'])
                total_correct += correct
                total_expected += expected
                
                # Test parameter extraction
                for doc in docs:
                    if 'parameters' not in doc:
                        doc['parameters'] = program.extract_parameters(doc.get('signature', ''))
                
                # Test formatting
                formatted = program.format_documentation(docs)
                
                # Store results for debugging
                artifacts[f'test_case_{i}'] = {
                    'expected_count': expected,
                    'found_count': correct,
                    'extracted_functions': [doc.get('name', 'unknown') for doc in docs],
                    'formatted_length': len(formatted)
                }
                
            except Exception as e:
                parsing_errors += 1
                artifacts[f'test_case_{i}_error'] = str(e)
        
        # Calculate metrics
        if total_expected > 0:
            metrics['accuracy'] = total_correct / total_expected
        
        metrics['completeness'] = min(1.0, total_correct / 20)  # Expect ~20 functions total
        metrics['robustness'] = max(0.0, 1.0 - (parsing_errors / len(test_cases)))
        metrics['parsing_errors'] = parsing_errors / len(test_cases)
        
        # Overall score - use 'combined_score' as primary metric for evolution
        metrics['combined_score'] = (
            metrics['accuracy'] * 0.4 +
            metrics['completeness'] * 0.3 +
            metrics['robustness'] * 0.3
        )
        
        # Add detailed feedback for the LLM
        artifacts['evaluation_feedback'] = generate_feedback(metrics, artifacts)
        
        # Return dictionary format for OpenEvolve compatibility
        return metrics
        
    except Exception as e:
        return {
            'accuracy': 0.0,
            'completeness': 0.0,
            'robustness': 0.0,
            'parsing_errors': 1.0,
            'combined_score': 0.0,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'stage': 'program_import'
        }


def get_test_cases() -> List[Dict[str, Any]]:
    """
    Get test cases with HTML content and expected results.
    
    These test cases include URLs that will be fetched by optillm's
    readurls plugin during evolution, providing the LLM with actual
    documentation structure.
    
    Returns:
        List of test cases with HTML content and expected results
    """
    return [
        {
            'name': 'json_module_docs',
            'html': '''
            <html>
            <body>
                <div class="section">
                    <h1>json — JSON encoder and decoder</h1>
                    <p>Source: https://docs.python.org/3/library/json.html</p>
                    
                    <dl class="function">
                        <dt class="sig sig-object py">
                            <span class="sig-name descname">dumps</span>
                            <span class="sig-paren">(</span>
                            <em class="sig-param">obj</em>,
                            <em class="sig-param">indent=None</em>
                            <span class="sig-paren">)</span>
                        </dt>
                        <dd>
                            <p>Serialize obj to a JSON formatted string.</p>
                        </dd>
                    </dl>
                    
                    <dl class="function">
                        <dt class="sig sig-object py">
                            <span class="sig-name descname">loads</span>
                            <span class="sig-paren">(</span>
                            <em class="sig-param">s</em>
                            <span class="sig-paren">)</span>
                        </dt>
                        <dd>
                            <p>Deserialize s to a Python object.</p>
                        </dd>
                    </dl>
                </div>
            </body>
            </html>
            ''',
            'expected': [
                {'name': 'dumps', 'params': ['obj', 'indent']},
                {'name': 'loads', 'params': ['s']}
            ]
        },
        {
            'name': 'requests_docs',
            'html': '''
            <html>
            <body>
                <div class="document">
                    <h1>Requests Documentation</h1>
                    <p>Refer to https://requests.readthedocs.io/en/latest/api/ for full API</p>
                    
                    <div class="function">
                        <h3>requests.get(url, params=None, **kwargs)</h3>
                        <p>Sends a GET request.</p>
                    </div>
                    
                    <div class="function">
                        <h3>requests.post(url, data=None, json=None, **kwargs)</h3>
                        <p>Sends a POST request.</p>
                    </div>
                </div>
            </body>
            </html>
            ''',
            'expected': [
                {'name': 'requests.get', 'params': ['url', 'params']},
                {'name': 'requests.post', 'params': ['url', 'data', 'json']}
            ]
        },
        {
            'name': 'beautifulsoup_docs',
            'html': '''
            <html>
            <body>
                <div class="section">
                    <h1>BeautifulSoup Documentation</h1>
                    <p>Documentation at https://www.crummy.com/software/BeautifulSoup/bs4/doc/</p>
                    
                    <code class="python">
                        <span class="name">BeautifulSoup</span>(<span class="param">markup</span>, <span class="param">parser</span>)
                    </code>
                    <p>Parse a string using a specified parser.</p>
                    
                    <code class="python">
                        <span class="name">find</span>(<span class="param">name</span>, <span class="param">attrs</span>=<span class="default">None</span>)
                    </code>
                    <p>Find the first matching tag.</p>
                    
                    <code class="python">
                        <span class="name">find_all</span>(<span class="param">name</span>, <span class="param">attrs</span>=<span class="default">None</span>, <span class="param">limit</span>=<span class="default">None</span>)
                    </code>
                    <p>Find all matching tags.</p>
                </div>
            </body>
            </html>
            ''',
            'expected': [
                {'name': 'BeautifulSoup', 'params': ['markup', 'parser']},
                {'name': 'find', 'params': ['name', 'attrs']},
                {'name': 'find_all', 'params': ['name', 'attrs', 'limit']}
            ]
        },
        {
            'name': 'edge_case_malformed',
            'html': '''
            <html>
            <body>
                <div class="weird-format">
                    <h2>Unusual Documentation Format</h2>
                    <p>This tests robustness - check https://example.com/weird-api-docs</p>
                    
                    <pre>
                    function_name(arg1, arg2=default_value)
                    Another description here
                    </pre>
                    
                    <table>
                        <tr>
                            <td>another_func()</td>
                            <td>Does something</td>
                        </tr>
                    </table>
                </div>
            </body>
            </html>
            ''',
            'expected': [
                {'name': 'function_name', 'params': ['arg1', 'arg2']},
                {'name': 'another_func', 'params': []}
            ]
        }
    ]


def evaluate_extraction(docs: List[Dict[str, Any]], expected: List[Dict[str, Any]]) -> tuple[int, int]:
    """
    Evaluate the accuracy of extracted documentation.
    
    Args:
        docs: Extracted documentation
        expected: Expected results
        
    Returns:
        Tuple of (correct_count, expected_count)
    """
    correct = 0
    expected_count = len(expected)
    
    for exp in expected:
        # Check if we found this function
        found = False
        for doc in docs:
            doc_name = doc.get('name', '').lower()
            exp_name = exp['name'].lower()
            
            if exp_name in doc_name or doc_name in exp_name:
                found = True
                # Check parameter extraction
                doc_params = doc.get('parameters', [])
                exp_params = exp.get('params', [])
                
                if len(doc_params) >= len(exp_params):
                    correct += 1
                else:
                    correct += 0.5  # Partial credit
                break
        
        if not found and docs:  # Only penalize if we extracted something
            pass  # No additional penalty
    
    return correct, expected_count


def generate_feedback(metrics: Dict[str, float], artifacts: Dict[str, Any]) -> str:
    """
    Generate detailed feedback for the LLM to improve the scraper.
    
    This feedback will be included in the evolution prompt to guide
    the LLM toward better solutions.
    
    Args:
        metrics: Evaluation metrics
        artifacts: Evaluation artifacts
        
    Returns:
        Detailed feedback string
    """
    feedback = []
    
    feedback.append("## Evaluation Feedback")
    feedback.append(f"Overall Score: {metrics['combined_score']:.2f}/1.0")
    feedback.append("")
    
    # Accuracy feedback
    if metrics['accuracy'] < 0.5:
        feedback.append("⚠️ **Low Accuracy**: The scraper is missing many expected functions.")
        feedback.append("Consider improving the HTML parsing logic to handle different documentation formats.")
        feedback.append("Look for patterns like <dl class='function'>, <div class='function'>, and <code> tags.")
    elif metrics['accuracy'] < 0.8:
        feedback.append("✅ **Good Accuracy**: Most functions are found, but some are missed.")
        feedback.append("Fine-tune the extraction logic for edge cases.")
    else:
        feedback.append("🎉 **Excellent Accuracy**: Function extraction is working well!")
    
    feedback.append("")
    
    # Completeness feedback
    if metrics['completeness'] < 0.5:
        feedback.append("⚠️ **Low Completeness**: Not extracting enough functions overall.")
        feedback.append("Increase the limit or improve the search scope.")
    
    # Robustness feedback
    if metrics['robustness'] < 0.8:
        feedback.append("⚠️ **Low Robustness**: The scraper fails on some HTML formats.")
        feedback.append("Add try-catch blocks and handle different documentation structures.")
        feedback.append("Consider multiple parsing strategies and fallback methods.")
    
    # Specific improvements
    feedback.append("")
    feedback.append("## Specific Improvements:")
    
    # Analyze test case results
    for key, value in artifacts.items():
        if key.startswith('test_case_') and isinstance(value, dict):
            if 'error' in key:
                feedback.append(f"- Fix error in {key}: {value}")
            elif value.get('found_count', 0) < value.get('expected_count', 0):
                feedback.append(f"- Improve extraction for {key}: found {value.get('found_count', 0)}/{value.get('expected_count', 0)} functions")
    
    # Documentation URL hints (these will be fetched by readurls plugin)
    feedback.append("")
    feedback.append("## Documentation References:")
    feedback.append("For improving parsing, refer to these documentation structures:")
    feedback.append("- Python docs: https://docs.python.org/3/library/json.html")
    feedback.append("- Requests docs: https://requests.readthedocs.io/en/latest/api/")
    feedback.append("- BeautifulSoup docs: https://www.crummy.com/software/BeautifulSoup/bs4/doc/")
    
    return '\n'.join(feedback)