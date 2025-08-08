
"""
Tests for O series model config check
"""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from openevolve.llm.openai import OpenAILLM


class TestOpenAILLM(unittest.TestCase):

    def setUp(self):
        self.model_cfg = SimpleNamespace(
            name="test-model",
            system_message="SYS",
            temperature=0.7,
            top_p=0.98,
            max_tokens=42,
            timeout=1,
            retries=0,
            retry_delay=0,
            api_base="https://api.openai.com/v1",
            api_key="fake-key",
            random_seed=123,
        )

        fake_choice = SimpleNamespace(message=SimpleNamespace(content=" OK"))
        fake_response = SimpleNamespace(choices=[fake_choice])

        self.fake_client = MagicMock()
        self.fake_client.chat.completions.create.return_value = fake_response

    def test_generate_happy_path(self):

        with patch("openevolve.llm.openai.openai.OpenAI", return_value=self.fake_client) as _:
            llm = OpenAILLM(self.model_cfg)

            result = asyncio.get_event_loop().run_until_complete(
                llm.generate("hello world")
            )

            self.assertEqual(result, " OK")

            called_kwargs = self.fake_client.chat.completions.create.call_args.kwargs
            msgs = called_kwargs["messages"]
            self.assertEqual(msgs[0]["role"], "system")
            self.assertEqual(msgs[0]["content"], "SYS")
            self.assertEqual(msgs[1]["role"], "user")
            self.assertEqual(msgs[1]["content"], "hello world")

    def test_max_completion_tokens_branch(self):
        self.model_cfg.name = "o4-mini"
        with patch("openevolve.llm.openai.openai.OpenAI", return_value=self.fake_client):
            llm = OpenAILLM(self.model_cfg)
            asyncio.get_event_loop().run_until_complete(llm.generate("foo"))

            called = self.fake_client.chat.completions.create.call_args.kwargs

            self.assertIn("max_completion_tokens", called)
            self.assertNotIn("max_tokens", called)

    def test_fallback_max_tokens_branch(self):

        self.model_cfg.api_base = "https://my.custom.endpoint"
        with patch("openevolve.llm.openai.openai.OpenAI", return_value=self.fake_client):
            llm = OpenAILLM(self.model_cfg)
            asyncio.get_event_loop().run_until_complete(llm.generate("bar"))

            called = self.fake_client.chat.completions.create.call_args.kwargs

            self.assertIn("max_tokens", called)
            self.assertNotIn("max_completion_tokens", called)
