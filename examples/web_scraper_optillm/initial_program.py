"""
Web scraper for extracting API documentation from HTML pages.

This initial implementation provides basic HTML parsing functionality
that will be evolved to handle complex documentation structures.

The LLM will have access to actual documentation pages through optillm's
readurls plugin, allowing it to understand the specific HTML structure
and improve the parsing logic accordingly.
"""

from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import re


# EVOLVE-BLOCK-START
def scrape_api_docs(html_content: str) -> List[Dict[str, any]]:
    """
    Extract API documentation from HTML content.
    
    Args:
        html_content: Raw HTML content of a documentation page
        
    Returns:
        List of dictionaries containing function documentation
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    functions = []
    
    # Try multiple approaches to find functions
    # 1. Look for code blocks
    code_blocks = soup.find_all('code')
    for block in code_blocks:
        text = block.get_text(strip=True)
        if '(' in text and ')' in text:
            functions.append({
                'name': text.split('(')[0].strip(),
                'signature': text,
                'description': 'No description found',
                'parameters': extract_parameters(text)
            })
    
    # 2. Look for function signatures in headers (h3)
    h3_blocks = soup.find_all('h3')
    for block in h3_blocks:
        text = block.get_text(strip=True)
        if '(' in text and ')' in text:
            functions.append({
                'name': text.split('(')[0].strip(),
                'signature': text,
                'description': 'No description found',
                'parameters': extract_parameters(text)
            })
    
    # 3. Look for dt elements with sig class
    dt_blocks = soup.find_all('dt', class_='sig')
    for block in dt_blocks:
        sig_name = block.find(class_='sig-name')
        if sig_name:
            name = sig_name.get_text(strip=True)
            functions.append({
                'name': name,
                'signature': block.get_text(strip=True),
                'description': 'No description found',
                'parameters': extract_parameters(block.get_text(strip=True))
            })
    
    return functions[:20]  # Return more functions


def extract_parameters(signature: str) -> List[Dict[str, str]]:
    """
    Extract parameter information from a function signature.
    
    Args:
        signature: Function signature string
        
    Returns:
        List of parameter dictionaries
    """
    params = []
    # Very basic parameter extraction
    match = re.search(r'\((.*?)\)', signature)
    if match:
        param_string = match.group(1)
        if param_string:
            param_parts = param_string.split(',')
            for part in param_parts:
                part = part.strip()
                if part:
                    params.append({
                        'name': part.split('=')[0].strip(),
                        'type': 'unknown',
                        'default': None,
                        'description': ''
                    })
    
    return params


def format_documentation(api_docs: List[Dict[str, any]]) -> str:
    """
    Format extracted documentation into a readable string.
    
    Args:
        api_docs: List of API documentation dictionaries
        
    Returns:
        Formatted documentation string
    """
    output = []
    for doc in api_docs:
        output.append(f"Function: {doc['name']}")
        output.append(f"Signature: {doc['signature']}")
        output.append(f"Description: {doc['description']}")
        
        if doc.get('parameters'):
            output.append("Parameters:")
            for param in doc['parameters']:
                output.append(f"  - {param['name']}: {param.get('description', 'No description')}")
        
        output.append("")  # Empty line between functions
    
    return '\n'.join(output)
# EVOLVE-BLOCK-END


# Example usage and test
if __name__ == "__main__":
    # Sample HTML for testing basic functionality
    sample_html = """
    <html>
    <body>
        <div class="function">
            <code>json.dumps(obj, indent=2)</code>
            <p>Serialize obj to a JSON formatted string.</p>
        </div>
        <div class="function">
            <code>json.loads(s)</code>
            <p>Deserialize s to a Python object.</p>
        </div>
    </body>
    </html>
    """
    
    docs = scrape_api_docs(sample_html)
    print(format_documentation(docs))
    print(f"\nExtracted {len(docs)} functions")