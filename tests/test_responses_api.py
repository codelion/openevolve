"""
Tests for OpenAI Responses API migration

This module tests the Responses API integration, including:
- Parameter conversion from Chat Completions format to Responses API format
- API selection logic based on endpoint and api_type config
- Response parsing differences between the two APIs
"""

import unittest
import asyncio
from unittest.mock import Mock, patch


class TestResponsesAPIParameterConversion(unittest.TestCase):
    """Test that Chat Completions parameters are correctly converted to Responses API format"""

    def setUp(self):
        """Set up test fixtures"""
        from openevolve.llm.openai import OpenAILLM
        
        self.model_cfg = Mock()
        self.model_cfg.name = "gpt-4o"
        self.model_cfg.system_message = "You are a helpful assistant"
        self.model_cfg.temperature = 0.7
        self.model_cfg.top_p = 0.95
        self.model_cfg.max_tokens = 4096
        self.model_cfg.timeout = 60
        self.model_cfg.retries = 3
        self.model_cfg.retry_delay = 5
        self.model_cfg.api_base = "https://api.openai.com/v1"
        self.model_cfg.api_key = "test-key"
        self.model_cfg.random_seed = None
        self.model_cfg.reasoning_effort = None
        self.model_cfg.api_type = "responses"  # Force Responses API

    def test_messages_to_input_conversion(self):
        """Test that messages array is converted to input parameter"""
        with patch('openai.OpenAI'):
            from openevolve.llm.openai import OpenAILLM
            llm = OpenAILLM(self.model_cfg)
            
            mock_response = Mock()
            mock_response.output_text = "Test response"
            llm.client.responses.create.return_value = mock_response
            
            chat_params = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "Be helpful"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"}
                ],
                "temperature": 0.7,
                "max_tokens": 100
            }
            
            asyncio.run(llm._call_api(chat_params))
            
            call_args = llm.client.responses.create.call_args.kwargs
            
            # System message should become instructions
            self.assertEqual(call_args["instructions"], "Be helpful")
            
            # Other messages should be in input array (excluding system)
            expected_input = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
            self.assertEqual(call_args["input"], expected_input)

    def test_max_tokens_conversion(self):
        """Test that max_tokens is converted to max_output_tokens"""
        with patch('openai.OpenAI'):
            from openevolve.llm.openai import OpenAILLM
            llm = OpenAILLM(self.model_cfg)
            
            mock_response = Mock()
            mock_response.output_text = "Test"
            llm.client.responses.create.return_value = mock_response
            
            # Test with max_tokens
            chat_params = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 500
            }
            
            asyncio.run(llm._call_api(chat_params))
            call_args = llm.client.responses.create.call_args.kwargs
            self.assertEqual(call_args["max_output_tokens"], 500)

    def test_max_completion_tokens_conversion(self):
        """Test that max_completion_tokens takes precedence over max_tokens"""
        with patch('openai.OpenAI'):
            from openevolve.llm.openai import OpenAILLM
            llm = OpenAILLM(self.model_cfg)
            
            mock_response = Mock()
            mock_response.output_text = "Test"
            llm.client.responses.create.return_value = mock_response
            
            # Test with max_completion_tokens (should take precedence)
            chat_params = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 500,
                "max_completion_tokens": 1000
            }
            
            asyncio.run(llm._call_api(chat_params))
            call_args = llm.client.responses.create.call_args.kwargs
            self.assertEqual(call_args["max_output_tokens"], 1000)

    def test_reasoning_effort_nested_format(self):
        """Test that reasoning_effort is converted to nested reasoning object"""
        with patch('openai.OpenAI'):
            from openevolve.llm.openai import OpenAILLM
            llm = OpenAILLM(self.model_cfg)
            
            mock_response = Mock()
            mock_response.output_text = "Test"
            llm.client.responses.create.return_value = mock_response
            
            chat_params = {
                "model": "o3-mini",
                "messages": [{"role": "user", "content": "Think hard"}],
                "reasoning_effort": "high"
            }
            
            asyncio.run(llm._call_api(chat_params))
            call_args = llm.client.responses.create.call_args.kwargs
            self.assertEqual(call_args["reasoning"], {"effort": "high"})

    def test_store_disabled(self):
        """Test that store is set to False for OpenEvolve use case"""
        with patch('openai.OpenAI'):
            from openevolve.llm.openai import OpenAILLM
            llm = OpenAILLM(self.model_cfg)
            
            mock_response = Mock()
            mock_response.output_text = "Test"
            llm.client.responses.create.return_value = mock_response
            
            chat_params = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}]
            }
            
            asyncio.run(llm._call_api(chat_params))
            call_args = llm.client.responses.create.call_args.kwargs
            self.assertFalse(call_args["store"])

    def test_sampling_params_preserved(self):
        """Test that temperature, top_p, and seed are preserved"""
        with patch('openai.OpenAI'):
            from openevolve.llm.openai import OpenAILLM
            llm = OpenAILLM(self.model_cfg)
            
            mock_response = Mock()
            mock_response.output_text = "Test"
            llm.client.responses.create.return_value = mock_response
            
            chat_params = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 0.5,
                "top_p": 0.9,
                "seed": 42
            }
            
            asyncio.run(llm._call_api(chat_params))
            call_args = llm.client.responses.create.call_args.kwargs
            self.assertEqual(call_args["temperature"], 0.5)
            self.assertEqual(call_args["top_p"], 0.9)
            self.assertEqual(call_args["seed"], 42)


class TestAPISelectionInOpenAILLM(unittest.TestCase):
    """Test the API selection logic in the OpenAILLM class"""

    def _create_model_cfg(self, api_base, api_type="auto"):
        """Helper to create a mock model config"""
        model_cfg = Mock()
        model_cfg.name = "gpt-4o"
        model_cfg.system_message = "test"
        model_cfg.temperature = 0.7
        model_cfg.top_p = 0.95
        model_cfg.max_tokens = 4096
        model_cfg.timeout = 60
        model_cfg.retries = 3
        model_cfg.retry_delay = 5
        model_cfg.api_base = api_base
        model_cfg.api_key = "test-key"
        model_cfg.random_seed = None
        model_cfg.reasoning_effort = None
        model_cfg.api_type = api_type
        return model_cfg

    def test_openai_endpoint_uses_responses_api(self):
        """Test that OpenAI endpoints use Responses API by default"""
        with patch('openai.OpenAI'):
            from openevolve.llm.openai import OpenAILLM
            
            llm = OpenAILLM(self._create_model_cfg("https://api.openai.com/v1"))
            self.assertTrue(llm.use_responses_api)
            
            llm = OpenAILLM(self._create_model_cfg("https://eu.api.openai.com/v1"))
            self.assertTrue(llm.use_responses_api)
            
            llm = OpenAILLM(self._create_model_cfg("https://apac.api.openai.com/v1"))
            self.assertTrue(llm.use_responses_api)

    def test_non_openai_endpoint_uses_chat_completions(self):
        """Test that non-OpenAI endpoints use Chat Completions API"""
        with patch('openai.OpenAI'):
            from openevolve.llm.openai import OpenAILLM
            
            llm = OpenAILLM(self._create_model_cfg("https://openrouter.ai/api/v1"))
            self.assertFalse(llm.use_responses_api)
            
            llm = OpenAILLM(self._create_model_cfg("http://localhost:8000/v1"))
            self.assertFalse(llm.use_responses_api)
            
            llm = OpenAILLM(self._create_model_cfg("https://generativelanguage.googleapis.com/v1beta/openai/"))
            self.assertFalse(llm.use_responses_api)

    def test_api_type_override_forces_responses(self):
        """Test that api_type='responses' forces Responses API"""
        with patch('openai.OpenAI'):
            from openevolve.llm.openai import OpenAILLM
            
            # Non-OpenAI endpoint with responses override
            llm = OpenAILLM(self._create_model_cfg("http://localhost:8000/v1", api_type="responses"))
            self.assertTrue(llm.use_responses_api)

    def test_api_type_override_forces_chat_completions(self):
        """Test that api_type='chat_completions' forces Chat Completions API"""
        with patch('openai.OpenAI'):
            from openevolve.llm.openai import OpenAILLM
            
            # OpenAI endpoint with chat_completions override
            llm = OpenAILLM(self._create_model_cfg("https://api.openai.com/v1", api_type="chat_completions"))
            self.assertFalse(llm.use_responses_api)


class TestResponsesAPIResponseParsing(unittest.TestCase):
    """Test that responses from both APIs are correctly parsed"""

    def _create_model_cfg(self, api_type):
        """Helper to create a mock model config"""
        model_cfg = Mock()
        model_cfg.name = "gpt-4o"
        model_cfg.system_message = "test"
        model_cfg.temperature = 0.7
        model_cfg.top_p = 0.95
        model_cfg.max_tokens = 4096
        model_cfg.timeout = 60
        model_cfg.retries = 3
        model_cfg.retry_delay = 5
        model_cfg.api_base = "https://api.openai.com/v1"
        model_cfg.api_key = "test-key"
        model_cfg.random_seed = None
        model_cfg.reasoning_effort = None
        model_cfg.api_type = api_type
        return model_cfg

    def test_responses_api_output_text(self):
        """Test that Responses API response.output_text is returned"""
        with patch('openai.OpenAI'):
            from openevolve.llm.openai import OpenAILLM
            llm = OpenAILLM(self._create_model_cfg("responses"))
            
            mock_response = Mock()
            mock_response.output_text = "This is from Responses API"
            llm.client.responses.create.return_value = mock_response
            
            result = asyncio.run(llm._call_api({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}]
            }))
            
            self.assertEqual(result, "This is from Responses API")

    def test_chat_completions_message_content(self):
        """Test that Chat Completions response.choices[0].message.content is returned"""
        with patch('openai.OpenAI'):
            from openevolve.llm.openai import OpenAILLM
            llm = OpenAILLM(self._create_model_cfg("chat_completions"))
            
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "This is from Chat Completions"
            llm.client.chat.completions.create.return_value = mock_response
            
            result = asyncio.run(llm._call_api({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}]
            }))
            
            self.assertEqual(result, "This is from Chat Completions")


class TestConfigWithAPIType(unittest.TestCase):
    """Test that api_type config option works correctly"""

    def test_api_type_default_is_none_for_model(self):
        """Test that api_type defaults to None in LLMModelConfig (inherits from parent)"""
        from openevolve.config import LLMModelConfig
        
        config = LLMModelConfig()
        self.assertIsNone(config.api_type)

    def test_api_type_default_is_auto_for_llm(self):
        """Test that api_type defaults to 'auto' in LLMConfig"""
        from openevolve.config import LLMConfig
        
        config = LLMConfig()
        self.assertEqual(config.api_type, "auto")

    def test_api_type_in_shared_config(self):
        """Test that api_type is propagated to models via shared config"""
        from openevolve.config import Config
        
        yaml_config = {
            "llm": {
                "api_base": "https://api.openai.com/v1",
                "api_key": "test-key",
                "api_type": "chat_completions",  # Force chat completions at LLM level
                "models": [{"name": "gpt-4o", "weight": 1.0}]
            }
        }
        
        config = Config.from_dict(yaml_config)
        
        # Model should inherit api_type from LLM config
        self.assertEqual(config.llm.models[0].api_type, "chat_completions")

    def test_api_type_model_override(self):
        """Test that model-level api_type overrides LLM-level"""
        from openevolve.config import Config
        
        yaml_config = {
            "llm": {
                "api_base": "https://api.openai.com/v1",
                "api_key": "test-key",
                "api_type": "chat_completions",
                "models": [
                    {"name": "gpt-4o", "weight": 1.0, "api_type": "responses"}  # Override
                ]
            }
        }
        
        config = Config.from_dict(yaml_config)
        
        # Model-level override should take precedence
        self.assertEqual(config.llm.models[0].api_type, "responses")


if __name__ == "__main__":
    unittest.main()
