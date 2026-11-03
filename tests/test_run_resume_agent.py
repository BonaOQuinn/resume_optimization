import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage

from prompts import EVIDENCE_PATH, LATEX_PATH
from run_resume_agent import run_resume_optimization
from tests.test_resume_workflow import RecordingChatModel
from tests.resume_fixture import LATEX, SOURCE, PASS


class RunResumeAgentTests(unittest.TestCase):
    def test_real_agent_tool_writes_are_exported_without_exporting_inputs(self):
        latex = LATEX
        model = RecordingChatModel(responses=[
            AIMessage(content="", tool_calls=[
                {"name": "write_file", "args": {"file_path": LATEX_PATH, "content": latex}, "id": "latex"},
                {"name": "write_file", "args": {"file_path": EVIDENCE_PATH, "content": "Evidence map"}, "id": "evidence"},
            ]),
            AIMessage(content="Draft written; independent validation follows."),
            AIMessage(content=PASS),
        ])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            saved = run_resume_optimization("Required: Python.", SOURCE, output, model=model)
            self.assertEqual(saved["latex"].read_text(encoding="utf-8"), latex + "\n")
            self.assertEqual(saved["evidence"].read_text(encoding="utf-8"), "Evidence map")
            self.assertIn("Draft written", saved["response"].read_text(encoding="utf-8"))
            self.assertTrue({"optimized_resume.tex", "resume_evidence.md", "response.md", "validation.json", "trace.json"}.issubset({p.name for p in output.iterdir()}))
            with self.assertRaises(FileExistsError):
                run_resume_optimization("Job", "Resume", output, model=model)

    def test_clarification_is_saved_without_fabricating_a_latex_output(self):
        # Each turn needs a distinct message ID, as with a real chat provider.
        model = RecordingChatModel(responses=[AIMessage(content="Please confirm your graduation date.") for _ in range(3)])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            saved = run_resume_optimization("Job", "Incomplete source", output, model=model)
            self.assertEqual(set(saved), {"response", "validation"})
            self.assertIn("graduation date", saved["response"].read_text(encoding="utf-8"))

    def test_rejected_agent_draft_is_corrected_in_a_fresh_invocation(self):
        bad = LATEX.replace("{article}", "{moderncv}")
        def write(content, call_id):
            return AIMessage(content="", tool_calls=[{"name": "write_file", "args": {"file_path": LATEX_PATH, "content": content}, "id": call_id}])
        model = RecordingChatModel(responses=[
            write(bad, "bad"), AIMessage(content="First draft written."),
            write(LATEX, "fixed"), AIMessage(content="Corrected draft written."),
            AIMessage(content=PASS),
        ])
        with tempfile.TemporaryDirectory() as directory:
            saved = run_resume_optimization("Python", SOURCE, Path(directory) / "run", model=model)
            self.assertEqual(saved["latex"].read_text(), LATEX + "\n")
            self.assertIn("Corrected draft", saved["response"].read_text())


if __name__ == "__main__":
    unittest.main()
