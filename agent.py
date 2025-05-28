# Code Generator Agent
import asyncio  # Added for retry sleep
import logging
import re
from typing import Optional, Dict, Any

from openai import (
    APIError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError
)

from config import settings
from core.interfaces import CodeGeneratorInterface

logger = logging.getLogger(__name__)

class CodeGeneratorAgent(CodeGeneratorInterface):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.pro_provider = settings.CUSTOM_PROVIDERS.get(settings.PRO_KEY)
        self.flash_provider = settings.CUSTOM_PROVIDERS.get(settings.FLASH_KEY)
        self.evaluation_provider = settings.CUSTOM_PROVIDERS.get(settings.EVALUATION_KEY)

        # Validate keys for non-ollama providers
        for key, provider in settings.CUSTOM_PROVIDERS.items():
            if provider['provider'] != 'ollama' and not provider.get('api_key'):
                raise ValueError(
                    f"{key} API_KEY not found in settings. Please set it in your .env file or config."
                )

        # Initialize OpenAI client (will override key/base_url per call)
        self.client = AsyncOpenAI(
            base_url=self.flash_provider['base_url'],
            api_key=self.flash_provider['api_key']
        )
        self.model_name = settings.LITELLM_DEFAULT_MODEL
        self.generation_config = {
            "temperature": settings.LITELLM_TEMPERATURE,
            "top_p": settings.LITELLM_TOP_P,
            "top_k": settings.LITELLM_TOP_K,
            "max_tokens": settings.LITELLM_MAX_TOKENS,
        }
        self.litellm_extra_params = {
            "base_url": settings.LITELLM_DEFAULT_BASE_URL,
        }
        logger.info(f"CodeGeneratorAgent initialized with model: {self.model_name}")

    async def generate_code(
        self,
        prompt: str,
        provider_name: Optional[str] = None,
        temperature: Optional[float] = None,
        output_format: str = "code"
    ) -> str:
        provider_name = provider_name or settings.FLASH_KEY
        provider_cfg = settings.CUSTOM_PROVIDERS[provider_name]
        provider_type = provider_cfg.get('provider', 'openai')

        logger.info(f"Using provider '{provider_name}' ({provider_type}) for code generation")

    async def generate_code(self, prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = None, output_format: str = "code", litellm_extra_params: Optional[Dict[str, Any]] = None) -> str:
        effective_model_name = model_name if model_name else self.model_name
        litellm_extra_params = litellm_extra_params or self.litellm_extra_params
        logger.info(f"Attempting to generate code using model: {effective_model_name}, output_format: {output_format}")
        
        if output_format == "diff":
            prompt += '''

Provide your changes as a sequence of diff blocks in the following format:
# Original code block to be found and replaced
=======
# New code block to replace the original

IMPORTANT DIFF GUIDELINES:
1. The SEARCH block MUST be an EXACT copy of code from the original - match whitespace, indentation, and line breaks precisely
2. Each SEARCH block should be large enough (3-5 lines minimum) to uniquely identify where the change should be made
3. Include context around the specific line(s) you want to change
4. Make multiple separate diff blocks if you need to change different parts of the code
5. For each diff, the SEARCH and REPLACE blocks must be complete, valid code segments
6. Pay special attention to matching the exact original indentation of the code in your SEARCH block, as this is crucial for correct application in environments sensitive to indentation (like Python).

Example of a good diff:
def calculate_sum(numbers):
    if not numbers:
        return 0
    result = 0
    for num in numbers:
        result += num
    return result

Make sure your diff can be applied correctly!
'''
        
        logger.debug(f"Received prompt for code generation (format: {output_format}):\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        
        current_generation_config = self.generation_config.copy()
        if temperature is not None:
            current_generation_config["temperature"] = temperature
            logger.debug(f"Using temperature override: {temperature}")

        retries = settings.API_MAX_RETRIES
        delay = settings.API_RETRY_DELAY_SECONDS

        for attempt in range(1, retries + 1):
            try:
                logger.debug(f"Attempt {attempt}/{retries} to '{provider_name}'")
                # Ollama path
                if provider_type == 'ollama':
                    url = provider_cfg['base_url'].rstrip('/') + '/api/chat'
                    payload = {
                        "model":    provider_cfg['model'],
                        "messages": [{"role": "user", "content": prompt}],
                        "stream":   False,
                    }
                    logger.debug(f"Ollama request URL: {url}")
                    logger.debug(f"Ollama payload: {payload}")

                    # Explicit httpx timeout: (connect, read)
                    timeout = httpx.Timeout(
                        connect=settings.API_TIMEOUT_CONNECT,  # e.g. 5.0
                        read=settings.API_TIMEOUT_READ,        # e.g. 60.0
                        write=settings.API_TIMEOUT_WRITE,      # you can set same as read or a separate env var
                        pool=settings.API_TIMEOUT_POOL         # likewise, or reuse read/write
                    )
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        try:
                            resp = await client.post(url, json=payload)
                            logger.debug(f"Ollama response status: {resp.status_code}")
                            resp.raise_for_status()
                        except httpx.ConnectTimeout:
                            logger.error("Connection to Ollama timed out.")
                            raise
                        except httpx.ReadTimeout:
                            logger.error("Read from Ollama timed out.")
                            raise
                        except httpx.HTTPStatusError as e:
                            logger.error(f"Ollama returned HTTP {e.response.status_code}: {e.response.text}")
                            raise

                        # Parse JSON (might still raise)
                        try:
                            data = resp.json()
                            logger.info(f"Ollama response JSON: {data}")
                        except ValueError as e:
                            logger.error(f"Invalid JSON from Ollama: {resp.text!r}")
                            raise

                        generated_text = data['message']['content']
                else:
                    # OpenAI-compatible path
                    self.client.api_key = provider_cfg['api_key']
                    self.client.base_url = provider_cfg['base_url']
                    logger.debug(f"OpenAI request to {self.client.base_url} with model {provider_cfg['model']}")
                    response = await self.client.chat.completions.create(
                        model=provider_cfg['model'],
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.generation_config.get("max_tokens"),
                        temperature=temperature or self.generation_config["temperature"],
                        top_p=self.generation_config["top_p"],
                    )
                    generated_text = response.choices[0].message.content
                    logger.debug(f"OpenAI generated text: {generated_text}")

                if not generated_text:
                    logger.error("No content returned from LLM.")
                    return ""

                if output_format == 'code':
                    cleaned = self._clean_llm_output(generated_text)
                    logger.debug(f"Cleaned code output: {cleaned}")
                    return cleaned
                else:
                    logger.debug(f"Raw diff output: {generated_text}")
                    return generated_text

            except (APIError, InternalServerError, APITimeoutError,
                    TimeoutError, RateLimitError, AuthenticationError,
                    BadRequestError, httpx.HTTPError) as e:
                logger.warning(f"Error on attempt {attempt}: {type(e).__name__} - {e}. Retrying in {delay}s.")
                if attempt < retries:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                logger.error(f"LLM Failed after {retries} attempts on effective_model '{effective_model_name}'.")
                raise
        
        logger.error(f"Code generation failed for model {effective_model_name} after all retries.")
        return ""

    def _clean_llm_output(self, raw_code: str) -> str:
        """
        Cleans the raw output from the LLM, typically removing markdown code fences.
        Example: ```python\ncode\n``` -> code
        """
        logger.debug(f"Attempting to clean raw LLM output. Input length: {len(raw_code)}")
        code = raw_code.strip()
        
        if code.startswith("```python") and code.endswith("```"):
            cleaned = code[len("```python"): -len("```")].strip()
            logger.debug("Cleaned Python markdown fences.")
            return cleaned
        elif code.startswith("```") and code.endswith("```"):
            cleaned = code[len("```"): -len("```")].strip()
            logger.debug("Cleaned generic markdown fences.")
            return cleaned
            
        logger.debug("No markdown fences found or standard cleaning applied to the stripped code.")
        return code

    def _apply_diff(self, parent_code: str, diff_text: str) -> str:
        """
        Applies a diff in the AlphaEvolve format to the parent code.
        Diff format:
        # Original code block



        
        Uses fuzzy matching to handle slight variations in whitespace and indentation.
        """
        logger.info("Attempting to apply diff.")
        logger.debug(f"Parent code length: {len(parent_code)}")
        logger.debug(f"Diff text:\n{diff_text}")

        modified_code = parent_code
        diff_pattern = re.compile(r"\s*?\n(.*?)\n", re.DOTALL)
        
                                                                                
                                                             
        replacements_made = []
        
        for match in diff_pattern.finditer(diff_text):
            search_block = match.group(1)
            replace_block = match.group(2)
            
                                                                        
            search_block_normalized = search_block.replace('\r\n', '\n').replace('\r', '\n').strip()
            
            try:
                                       
                if search_block_normalized in modified_code:
                    logger.debug(f"Found exact match for SEARCH block")
                    modified_code = modified_code.replace(search_block_normalized, replace_block, 1)
                    logger.debug(f"Applied one diff block. SEARCH:\n{search_block_normalized}\nREPLACE:\n{replace_block}")
                else:
                                                                                 
                    normalized_search = re.sub(r'\s+', ' ', search_block_normalized)
                    normalized_code = re.sub(r'\s+', ' ', modified_code)
                    
                    if normalized_search in normalized_code:
                        logger.debug(f"Found match after whitespace normalization")
                                                                        
                        start_pos = normalized_code.find(normalized_search)
                        
                                                                          
                        original_pos = 0
                        norm_pos = 0
                        
                        while norm_pos < start_pos and original_pos < len(modified_code):
                            if not modified_code[original_pos].isspace() or (
                                original_pos > 0 and 
                                modified_code[original_pos].isspace() and 
                                not modified_code[original_pos-1].isspace()
                            ):
                                norm_pos += 1
                            original_pos += 1
                        
                                               
                        end_pos = original_pos
                        remaining_chars = len(normalized_search)
                        
                        while remaining_chars > 0 and end_pos < len(modified_code):
                            if not modified_code[end_pos].isspace() or (
                                end_pos > 0 and 
                                modified_code[end_pos].isspace() and 
                                not modified_code[end_pos-1].isspace()
                            ):
                                remaining_chars -= 1
                            end_pos += 1
                        
                                                                                        
                        overlap = False
                        for start, end in replacements_made:
                            if (start <= original_pos <= end) or (start <= end_pos <= end):
                                overlap = True
                                break
                        
                        if not overlap:
                                                               
                            actual_segment = modified_code[original_pos:end_pos]
                            logger.debug(f"Replacing segment:\n{actual_segment}\nWith:\n{replace_block}")
                            
                                                 
                            modified_code = modified_code[:original_pos] + replace_block + modified_code[end_pos:]
                            
                                                     
                            replacements_made.append((original_pos, original_pos + len(replace_block)))
                        else:
                            logger.warning(f"Diff application: Skipping overlapping replacement")
                    else:
                                                               
                        search_lines = search_block_normalized.splitlines()
                        parent_lines = modified_code.splitlines()
                        
                                                                      
                        if len(search_lines) >= 3:
                                                                  
                            first_line = search_lines[0].strip()
                            last_line = search_lines[-1].strip()
                            
                            for i, line in enumerate(parent_lines):
                                if first_line in line.strip() and i + len(search_lines) <= len(parent_lines):
                                                                     
                                    if last_line in parent_lines[i + len(search_lines) - 1].strip():
                                                                                       
                                        matched_segment = '\n'.join(parent_lines[i:i + len(search_lines)])
                                        
                                                              
                                        modified_code = '\n'.join(
                                            parent_lines[:i] + 
                                            replace_block.splitlines() + 
                                            parent_lines[i + len(search_lines):]
                                        )
                                        logger.debug(f"Applied line-by-line match. SEARCH:\n{matched_segment}\nREPLACE:\n{replace_block}")
                                        break
                            else:
                                logger.warning(f"Diff application: SEARCH block not found even with line-by-line search:\n{search_block_normalized}")
                        else:
                            logger.warning(f"Diff application: SEARCH block not found in current code state:\n{search_block_normalized}")
            except re.error as e:
                logger.error(f"Regex error during diff application: {e}")
                continue
            except Exception as e:
                logger.error(f"Error during diff application: {e}", exc_info=True)
                continue
        
        if modified_code == parent_code and diff_text.strip():
             logger.warning("Diff text was provided, but no changes were applied. Check SEARCH blocks/diff format.")
        elif modified_code != parent_code:
             logger.info("Diff successfully applied, code has been modified.")
        else:
             logger.info("No diff text provided or diff was empty, code unchanged.")
             
        return modified_code

    async def execute(self, prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = None, output_format: str = "code", parent_code_for_diff: Optional[str] = None, litellm_extra_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generic execution method.
        If output_format is 'diff', it generates a diff and applies it to parent_code_for_diff.
        Otherwise, it generates full code.
        """
        logger.debug(f"CodeGeneratorAgent.execute called. Output format: {output_format}")
        
        generated_output = await self.generate_code(
            prompt=prompt, 
            model_name=model_name, 
            temperature=temperature,
            output_format=output_format,
            litellm_extra_params=litellm_extra_params
        )

        if output_format == "diff":
            if not parent_code_for_diff:
                logger.error("Output format is 'diff' but no parent_code_for_diff provided. Returning raw diff.")
                return generated_output 
            
            if not generated_output.strip():
                 logger.info("Generated diff is empty. Returning parent code.")
                 return parent_code_for_diff

            try:
                logger.info("Applying generated diff to parent code.")
                modified_code = self._apply_diff(parent_code_for_diff, generated_output)
                return modified_code
            except Exception as e:
                logger.error(f"Error applying diff: {e}. Returning raw diff text.", exc_info=True)
                return generated_output
        else:         
            return generated_output

                                                 
if __name__ == '__main__':
    import asyncio
    logging.basicConfig(level=logging.DEBUG)
    from unittest.mock import Mock # Added for mocking
    
    async def test_diff_application():
        agent = CodeGeneratorAgent()
        parent = """Line 1
Line 2 to be replaced
Line 3
Another block
To be changed
End of block
Final line"""

        diff = """Some preamble text from LLM...
Line 2 to be replaced

Some other text...

Another block
To be changed
End of block
Trailing text..."""
        expected_output = """Line 1
Line 2 has been successfully replaced
Line 3
This
Entire
Block
Is New
Final line"""
        
        print("--- Testing _apply_diff directly ---")
        result = agent._apply_diff(parent, diff)
        print("Result of diff application:")
        print(result)
        assert result.strip() == expected_output.strip(), f"Direct diff application failed.\nExpected:\n{expected_output}\nGot:\n{result}"
        print("_apply_diff test passed.")

        print("\n--- Testing execute with output_format='diff' ---")
        async def mock_generate_code(prompt, model_name, temperature, output_format, litellm_extra_params=None): # Added litellm_extra_params
            return diff
        
        agent.generate_code = mock_generate_code 
        
        result_execute_diff = await agent.execute(
            prompt="doesn't matter for this mock", 
            parent_code_for_diff=parent,
            output_format="diff",
            litellm_extra_params={"example_param": "example_value"} # Added for testing
        )
        print("Result of execute with diff:")
        print(result_execute_diff)
        assert result_execute_diff.strip() == expected_output.strip(), f"Execute with diff failed.\nExpected:\n{expected_output}\nGot:\n{result_execute_diff}"
        print("Execute with diff test passed.")


    async def test_generation():
        agent = CodeGeneratorAgent()
        
        test_prompt_full_code = "Write a Python function that takes two numbers and returns their sum."
        
        # Mock litellm.acompletion for full code generation test
        original_acompletion = litellm.acompletion
        async def mock_litellm_acompletion(*args, **kwargs):
            mock_response = Mock()
            mock_message = Mock()
            mock_message.content = "def mock_function():\n  return 'mocked_code'"
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = mock_message
            return mock_response
        
        litellm.acompletion = mock_litellm_acompletion
        
        try:
            generated_full_code = await agent.execute(test_prompt_full_code, temperature=0.6, output_format="code")
            print("\n--- Generated Full Code (via execute) ---")
            print(generated_full_code)
            print("----------------------")
            assert "def mock_function" in generated_full_code, "Full code generation with mock seems to have failed."
        finally:
            litellm.acompletion = original_acompletion

        parent_code_for_llm_diff = '''
def greet(name):
    return f"Hello, {name}!"

def process_data(data):
    # TODO: Implement data processing
    return data * 2 # Simple placeholder
'''
        test_prompt_diff_gen = f'''
Current code:
```python
{parent_code_for_llm_diff}
```
Task: Modify the `process_data` function to add 5 to the result instead of multiplying by 2.
Also, change the greeting in `greet` to "Hi, {name}!!!".
'''
                                                                            
                                                           
                                          
                              
                                   
                                                           
           
                                                                       
                                           
                                         
                                                                                                               
                                                                                                           
        
        async def mock_generate_empty_diff(prompt, model_name, temperature, output_format):
            return "  \n  " 
        
        original_generate_code = agent.generate_code 
        agent.generate_code = mock_generate_empty_diff
        
        print("\n--- Testing execute with empty diff from LLM ---")
        result_empty_diff = await agent.execute(
            prompt="doesn't matter",
            parent_code_for_diff=parent_code_for_llm_diff,
            output_format="diff"
        )
        assert result_empty_diff == parent_code_for_llm_diff, "Empty diff should return parent code."
        print("Execute with empty diff test passed.")
        agent.generate_code = original_generate_code

    async def main_tests():
        await test_diff_application()
                                                                                     
        print("\nAll selected local tests in CodeGeneratorAgent passed.")

    asyncio.run(main_tests())
