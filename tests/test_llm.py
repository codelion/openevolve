"""
Tests for LLM implementations
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from openevolve.config import LLMConfig
from openevolve.llm.anthropic import AnthropicLLM
from openevolve.llm.openai import OpenAILLM


class TestLLMImplementations(unittest.TestCase):
    """Tests for LLM implementations"""

    def setUp(self):
        """Set up test configuration"""
        self.config = LLMConfig(
            primary_model="test-model",
            api_key="test-key",
            api_base="https://test.api",
        )

    @patch("anthropic.Anthropic")
    async def test_anthropic_llm_generate(self, mock_anthropic):
        """Test Anthropic LLM generate method"""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response")]
        mock_anthropic.return_value.messages.create.return_value = mock_response

        # Create LLM instance
        llm = AnthropicLLM(self.config)

        # Test generate
        response = await llm.generate("Test prompt")
        self.assertEqual(response, "Test response")

        # Verify API call
        mock_anthropic.return_value.messages.create.assert_called_once()
        call_args = mock_anthropic.return_value.messages.create.call_args[1]
        self.assertEqual(call_args["model"], "test-model")
        self.assertEqual(call_args["messages"][0]["role"], "user")
        self.assertEqual(call_args["messages"][0]["content"], "Test prompt")

    @patch("anthropic.Anthropic")
    async def test_anthropic_llm_generate_with_context(self, mock_anthropic):
        """Test Anthropic LLM generate_with_context method"""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response")]
        mock_anthropic.return_value.messages.create.return_value = mock_response

        # Create LLM instance
        llm = AnthropicLLM(self.config)

        # Test generate_with_context
        messages = [
            {"role": "user", "content": "Test message 1"},
            {"role": "assistant", "content": "Test response 1"},
            {"role": "user", "content": "Test message 2"},
        ]
        response = await llm.generate_with_context("Test system", messages)
        self.assertEqual(response, "Test response")

        # Verify API call
        mock_anthropic.return_value.messages.create.assert_called_once()
        call_args = mock_anthropic.return_value.messages.create.call_args[1]
        self.assertEqual(call_args["model"], "test-model")
        self.assertEqual(call_args["system"], "Test system")
        self.assertEqual(len(call_args["messages"]), 3)
        self.assertEqual(call_args["messages"][0]["role"], "user")
        self.assertEqual(call_args["messages"][0]["content"], "Test message 1")

    @patch("openai.OpenAI")
    async def test_openai_llm_generate(self, mock_openai):
        """Test OpenAI LLM generate method"""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        # Create LLM instance
        llm = OpenAILLM(self.config)

        # Test generate
        response = await llm.generate("Test prompt")
        self.assertEqual(response, "Test response")

        # Verify API call
        mock_openai.return_value.chat.completions.create.assert_called_once()
        call_args = mock_openai.return_value.chat.completions.create.call_args[1]
        self.assertEqual(call_args["model"], "test-model")
        self.assertEqual(call_args["messages"][0]["role"], "user")
        self.assertEqual(call_args["messages"][0]["content"], "Test prompt")

    @patch("openai.OpenAI")
    async def test_openai_llm_generate_with_context(self, mock_openai):
        """Test OpenAI LLM generate_with_context method"""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        # Create LLM instance
        llm = OpenAILLM(self.config)

        # Test generate_with_context
        messages = [
            {"role": "user", "content": "Test message 1"},
            {"role": "assistant", "content": "Test response 1"},
            {"role": "user", "content": "Test message 2"},
        ]
        response = await llm.generate_with_context("Test system", messages)
        self.assertEqual(response, "Test response")

        # Verify API call
        mock_openai.return_value.chat.completions.create.assert_called_once()
        call_args = mock_openai.return_value.chat.completions.create.call_args[1]
        self.assertEqual(call_args["model"], "test-model")
        self.assertEqual(call_args["messages"][0]["role"], "system")
        self.assertEqual(call_args["messages"][0]["content"], "Test system")
        self.assertEqual(len(call_args["messages"]), 4)  # system + 3 messages

    def test_llm_config_model_detection(self):
        """Test LLM configuration model type detection"""
        # Test OpenAI model
        config = LLMConfig(primary_model="gpt-4")
        self.assertEqual(config.api_base, "https://api.openai.com/v1")

        # Test Claude model
        config = LLMConfig(primary_model="claude-3-sonnet")
        self.assertEqual(config.api_base, "https://api.anthropic.com/v1")

        # Test custom API base
        config = LLMConfig(
            primary_model="claude-3-sonnet",
            api_base="https://custom.api",
        )
        self.assertEqual(config.api_base, "https://custom.api")


if __name__ == "__main__":
    unittest.main()
