import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage

from prompts import EVIDENCE_PATH, LATEX_PATH
from run_resume_agent import run_resume_optimization
from tests.test_resume_workflow import RecordingChatModel


class RunResumeAgentTests(unittest.TestCase):
    def test_real_agent_tool_writes_are_exported_without_exporting_inputs(self):
        latex = "\\documentclass{article}\n\\begin{document}Source-supported draft\\end{document}"
        model = RecordingChatModel(responses=[
            AIMessage(content="", tool_calls=[
                {"name": "write_file", "args": {"file_path": LATEX_PATH, "content": latex}, "id": "latex"},
                {"name": "write_file", "args": {"file_path": EVIDENCE_PATH, "content": "Evidence map"}, "id": "evidence"},
            ]),
            AIMessage(content="Draft written; compilation has not been performed."),
        ])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            saved = run_resume_optimization("Required: Python.", "Original source facts.", output, model=model)
            self.assertEqual(saved["latex"].read_text(encoding="utf-8"), latex)
            self.assertEqual(saved["evidence"].read_text(encoding="utf-8"), "Evidence map")
            self.assertIn("compilation", saved["response"].read_text(encoding="utf-8"))
            self.assertEqual({p.name for p in output.iterdir()},
                             {"optimized_resume.tex", "resume_evidence.md", "response.md"})
            with self.assertRaises(FileExistsError):
                run_resume_optimization("Job", "Resume", output, model=model)

    def test_clarification_is_saved_without_fabricating_a_latex_output(self):
        model = RecordingChatModel(responses=[AIMessage(content="Please confirm your graduation date.")])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            saved = run_resume_optimization("Job", "Incomplete source", output, model=model)
            self.assertEqual(set(saved), {"response"})
            self.assertIn("graduation date", saved["response"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
