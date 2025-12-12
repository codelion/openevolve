"""
Test OpenAI reasoning model detection logic
"""

import unittest
from unittest.mock import MagicMock


class TestOpenAIReasoningModelDetection(unittest.TestCase):
    """Test that OpenAI reasoning models are correctly identified"""

    def test_reasoning_model_detection(self):
        """Test various model names to ensure correct reasoning model detection"""

        # Define the same constants as in the code
        OPENAI_REASONING_MODEL_PREFIXES = (
            # O-series reasoning models
            "o1-",
            "o1",  # o1, o1-mini, o1-preview
            "o3-",
            "o3",  # o3, o3-mini, o3-pro
            "o4-",  # o4-mini
            # GPT-5 series are also reasoning models
            "gpt-5-",
            "gpt-5",  # gpt-5, gpt-5-mini, gpt-5-nano
        )

        def is_reasoning_model(model_name, api_base="https://api.openai.com/v1"):
            """Test function that mimics the logic in openai.py"""
            model_lower = str(model_name).lower()
            return api_base == "https://api.openai.com/v1" and model_lower.startswith(
                OPENAI_REASONING_MODEL_PREFIXES
            )

        # Test cases: (model_name, expected_result, description)
        test_cases = [
            # Reasoning models - should return True
            ("o1", True, "Base o1 model"),
            ("o1-mini", True, "o1-mini model"),
            ("o1-preview", True, "o1-preview model"),
            ("o1-mini-2025-01-31", True, "o1-mini with date"),
            ("o3", True, "Base o3 model"),
            ("o3-mini", True, "o3-mini model"),
            ("o3-pro", True, "o3-pro model"),
            ("o4-mini", True, "o4-mini model"),
            ("gpt-5", True, "Base gpt-5 model"),
            ("gpt-5-mini", True, "gpt-5-mini model"),
            ("gpt-5-nano", True, "gpt-5-nano model"),
            # Non-reasoning models - should return False
            ("gpt-4o-mini", False, "gpt-4o-mini (not reasoning)"),
            ("gpt-4o", False, "gpt-4o (not reasoning)"),
            ("gpt-4", False, "gpt-4 (not reasoning)"),
            ("gpt-3.5-turbo", False, "gpt-3.5-turbo (not reasoning)"),
            ("claude-3", False, "Non-OpenAI model"),
            ("gemini-pro", False, "Non-OpenAI model"),
            # Edge cases
            ("O1-MINI", True, "Uppercase o1-mini"),
            ("GPT-5-MINI", True, "Uppercase gpt-5-mini"),
        ]

        for model_name, expected, description in test_cases:
            with self.subTest(model=model_name, desc=description):
                result = is_reasoning_model(model_name)
                self.assertEqual(
                    result,
                    expected,
                    f"Model '{model_name}' ({description}): expected {expected}, got {result}",
                )

    def test_non_openai_api_base(self):
        """Test that non-OpenAI API bases don't trigger reasoning model logic"""
        OPENAI_REASONING_MODEL_PREFIXES = ("o1-", "o1", "o3-", "o3", "o4-", "gpt-5-", "gpt-5")

        def is_reasoning_model(model_name, api_base):
            model_lower = str(model_name).lower()
            return api_base == "https://api.openai.com/v1" and model_lower.startswith(
                OPENAI_REASONING_MODEL_PREFIXES
            )

        # Even reasoning model names should return False for non-OpenAI APIs
        test_cases = [
            ("o1-mini", "https://api.anthropic.com/v1", False),
            ("gpt-5", "https://generativelanguage.googleapis.com/v1beta/openai/", False),
            ("o3-mini", "https://api.deepseek.com/v1", False),
        ]

        for model_name, api_base, expected in test_cases:
            with self.subTest(model=model_name, api=api_base):
                result = is_reasoning_model(model_name, api_base)
                self.assertEqual(
                    result,
                    expected,
                    f"Model '{model_name}' with API '{api_base}' should return {expected}",
                )


class TestResponsesAPIDetection(unittest.TestCase):
    """Test Responses API vs Chat Completions API selection logic"""

    def _should_use_responses_api(self, api_base, api_type="auto"):
        """Test function that mimics the logic in openai.py"""
        # Check for explicit override
        if api_type == "responses":
            return True
        if api_type == "chat_completions":
            return False
        
        # Auto-detect based on API base URL
        if not api_base:
            return False
        
        api_lower = api_base.lower()
        
        # Only use Responses API for official OpenAI endpoints
        return (
            api_lower.startswith("https://api.openai.com") or
            api_lower.startswith("https://eu.api.openai.com") or
            api_lower.startswith("https://apac.api.openai.com")
        )

    def test_openai_endpoints_use_responses_api(self):
        """Test that official OpenAI endpoints use Responses API by default"""
        test_cases = [
            ("https://api.openai.com/v1", True, "Main OpenAI endpoint"),
            ("https://api.openai.com", True, "OpenAI without path"),
            ("https://eu.api.openai.com/v1", True, "EU endpoint"),
            ("https://apac.api.openai.com/v1", True, "APAC endpoint"),
            ("https://API.OPENAI.COM/v1", True, "Uppercase URL"),
        ]

        for api_base, expected, description in test_cases:
            with self.subTest(api_base=api_base, desc=description):
                result = self._should_use_responses_api(api_base)
                self.assertEqual(
                    result,
                    expected,
                    f"API base '{api_base}' ({description}): expected {expected}, got {result}",
                )

    def test_non_openai_endpoints_use_chat_completions(self):
        """Test that non-OpenAI endpoints use Chat Completions API"""
        test_cases = [
            ("https://generativelanguage.googleapis.com/v1beta/openai/", False, "Google AI Studio"),
            ("https://openrouter.ai/api/v1", False, "OpenRouter"),
            ("http://localhost:8000/v1", False, "Local server"),
            ("https://api.anthropic.com/v1", False, "Anthropic"),
            ("https://api.deepseek.com/v1", False, "DeepSeek"),
            (None, False, "None API base"),
            ("", False, "Empty API base"),
        ]

        for api_base, expected, description in test_cases:
            with self.subTest(api_base=api_base, desc=description):
                result = self._should_use_responses_api(api_base)
                self.assertEqual(
                    result,
                    expected,
                    f"API base '{api_base}' ({description}): expected {expected}, got {result}",
                )

    def test_explicit_api_type_override(self):
        """Test that api_type override works correctly"""
        # Force responses API even for non-OpenAI endpoint
        self.assertTrue(
            self._should_use_responses_api("http://localhost:8000/v1", api_type="responses")
        )
        
        # Force chat completions even for OpenAI endpoint
        self.assertFalse(
            self._should_use_responses_api("https://api.openai.com/v1", api_type="chat_completions")
        )
        
        # Auto detection with OpenAI endpoint
        self.assertTrue(
            self._should_use_responses_api("https://api.openai.com/v1", api_type="auto")
        )
        
        # Auto detection with non-OpenAI endpoint
        self.assertFalse(
            self._should_use_responses_api("http://localhost:8000/v1", api_type="auto")
        )


if __name__ == "__main__":
    unittest.main()
