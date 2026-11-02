import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage

import agent as resume_agent
from tests.test_resume_workflow import RecordingChatModel


ROOT = Path(__file__).resolve().parents[1]


class ChatInitializationTests(unittest.TestCase):
    def test_environment_selection_and_explicit_override_keep_ollama_tags(self):
        with patch.dict(os.environ, {"RESUME_CHAT_MODEL": "ollama:configured-model:8b"}), \
             patch.object(resume_agent, "init_chat_model") as initialize:
            resume_agent.initialize_chat_model()
            initialize.assert_called_once_with("ollama:configured-model:8b")
            initialize.reset_mock()
            resume_agent.initialize_chat_model("ollama:explicit-model:3b")
            initialize.assert_called_once_with("ollama:explicit-model:3b")

    def test_missing_or_malformed_selection_fails_before_provider_initialization(self):
        with patch.dict(os.environ, {"RESUME_CHAT_MODEL": ""}), \
             patch.object(resume_agent, "init_chat_model") as initialize:
            for selected in (None, "", "  ", "all-MiniLM-L6-v2", "ollama:", ":model"):
                with self.subTest(selected=selected), self.assertRaises(ValueError):
                    resume_agent.initialize_chat_model(selected)
            initialize.assert_not_called()

    def test_configured_model_creates_a_working_graph_without_external_inference(self):
        model = RecordingChatModel(responses=[
            AIMessage(content="Please confirm the ambiguous employment date.")
        ])
        with patch.dict(os.environ, {"RESUME_CHAT_MODEL": "ollama:configured-model"}), \
             patch.object(resume_agent, "init_chat_model", return_value=model) as initialize:
            compiled = resume_agent.create_resume_agent(output_mode="latex_only")
            result = compiled.invoke(resume_agent.prepare_resume_request(
                "Required: Python.", "Built Python services; employment date needs confirmation."
            ))
        initialize.assert_called_once_with("ollama:configured-model")
        self.assertEqual(result["messages"][-1].content,
                         "Please confirm the ambiguous employment date.")
        self.assertTrue(any(
            "Education is always" in str(message.content)
            for messages in model.seen for message in messages
        ))

    def test_real_ollama_client_and_agent_construct_without_requesting_inference(self):
        with patch("httpx.Client.send", side_effect=AssertionError("Unexpected HTTP request")), \
             patch("httpx.AsyncClient.send", side_effect=AssertionError("Unexpected HTTP request")):
            model = resume_agent.initialize_chat_model("ollama:offline-test-model")
            compiled = resume_agent.create_resume_agent(model=model)
        self.assertEqual(model.model, "offline-test-model")
        self.assertTrue(callable(compiled.invoke))

    def test_cli_help_works_without_model_configuration(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "agent.py"), "--help"],
            cwd=ROOT, env={**os.environ, "RESUME_CHAT_MODEL": ""},
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model", result.stdout)


if __name__ == "__main__":
    unittest.main()
