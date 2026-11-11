import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage

from direct_resume import extract_complete_latex, run_direct_resume_optimization
from tests.test_resume_workflow import RecordingChatModel


from tests.resume_fixture import LATEX, SOURCE, PASS


class DirectResumeTests(unittest.TestCase):
    def test_one_call_gets_all_inputs_and_saves_complete_latex(self):
        model = RecordingChatModel(responses=[AIMessage(content=LATEX), AIMessage(content=PASS)])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            saved = run_direct_resume_optimization("Required: Python.", SOURCE, output, model=model)
            self.assertEqual(saved["latex"].read_text(encoding="utf-8"), LATEX + "\n")
            self.assertEqual(saved["response"].read_text(encoding="utf-8"), LATEX)
            self.assertTrue({"response.md", "optimized_resume.tex", "validation.json"}.issubset({p.name for p in output.iterdir()}))
            with self.assertRaises(FileExistsError):
                run_direct_resume_optimization("Job", "Resume", output, model=model)
        self.assertEqual(len(model.seen), 2)  # Writer plus independent grounding review.
        system, user = model.seen[0]
        self.assertIn("You have no tools or subagents", system.content)
        self.assertIn("Education is always the first", system.content)
        self.assertIn("Never invent", system.content)
        inputs = json.loads(user.content.split("\n", 1)[1])
        self.assertEqual(inputs["source_resume"], SOURCE)
        self.assertEqual(inputs["job_posting"], "Required: Python.")

    def test_fenced_latex_and_separate_thinking_are_accepted(self):
        self.assertEqual(extract_complete_latex("```latex\n" + LATEX + "\n```"), LATEX + "\n")
        self.assertEqual(extract_complete_latex("<think>Plan the draft.</think>\n" + LATEX), LATEX + "\n")

    def test_review_offers_clarifications_and_truncated_output_are_not_drafts(self):
        responses = [
            "I'm ready to review your LaTeX resume. Please paste the code.",
            "Please confirm your graduation date.",
            LATEX.replace(r"\end{document}", ""),
            LATEX + "\n" + LATEX,
            "Here is how to write it:\n" + LATEX,
            "<think>" + LATEX,
        ]
        for response in responses:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as directory:
                model = RecordingChatModel(responses=[AIMessage(content=response)])
                saved = run_direct_resume_optimization("Job", "Source facts", Path(directory) / "run", model=model)
                self.assertEqual(set(saved), {"response", "validation"})
                self.assertEqual(saved["response"].read_text(encoding="utf-8"), response)
                self.assertFalse((saved["response"].parent / "optimized_resume.tex").exists())

    def test_corrections_revise_the_previous_draft_and_keep_earlier_feedback(self):
        # A real run failed because each retry regenerated from scratch with only
        # the latest feedback, so attempt 3 repeated the error fixed in attempt 2.
        first = '{"verdict":"REVISE","issues":["Keep the source attribution in bullet 2."]}'
        second = '{"verdict":"REVISE","issues":["Say table, not tables."]}'
        model = RecordingChatModel(responses=[
            AIMessage(content=LATEX), AIMessage(content=first),
            AIMessage(content=LATEX), AIMessage(content=second),
            AIMessage(content=LATEX), AIMessage(content=PASS)])
        with tempfile.TemporaryDirectory() as directory:
            saved = run_direct_resume_optimization("Python", SOURCE, Path(directory) / "run", model=model)
            self.assertIn("latex", saved)
        second_try, third_try = model.seen[2], model.seen[4]
        self.assertEqual(second_try[-2].content, LATEX)  # The rejected draft is revised in place.
        self.assertIn("Keep the source attribution", second_try[-1].content)
        self.assertEqual(third_try[-2].content, LATEX)
        self.assertIn("Say table, not tables.", third_try[-1].content)
        self.assertIn("Keep the source attribution", third_try[-1].content)
        self.assertIn("earlier attempts", third_try[-1].content)

    def test_missing_inputs_fail_before_model_initialization(self):
        with patch("agent.initialize_chat_model") as initialize:
            for job, resume in (("", "Resume"), ("Job", " "), (None, "Resume")):
                with self.assertRaises(ValueError):
                    run_direct_resume_optimization(job, resume, Path("unused-test-output"))
            initialize.assert_not_called()

    def test_cli_direct_mode_skips_the_vector_store_and_agent(self):
        import run_resume_agent as runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, resume = root / "job.md", root / "resume.tex"
            job.write_text("Job", encoding="utf-8")
            resume.write_text("Resume", encoding="utf-8")
            output = root / "run"
            output.mkdir()
            response, latex = output / "response.md", output / "optimized_resume.tex"
            response.write_text(LATEX, encoding="utf-8")
            latex.write_text(LATEX, encoding="utf-8")
            # Let the runner create/check a fresh output directory.
            requested = root / "new-run"
            with patch("sys.argv", ["run_resume_agent.py", "--direct", "--job-file", str(job),
                                    "--resume-file", str(resume), "--output-dir", str(requested)]), \
                 patch.object(runner, "run_direct_resume_optimization", return_value={"response": response, "latex": latex, "validation": output / "validation.json"}) as direct, \
                 patch.object(runner, "get_resume_vector_store") as store, \
                 patch.object(runner, "run_resume_optimization") as workflow, \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            direct.assert_called_once_with("Job", "Resume", requested, model=None, require_pdf=False)
            store.assert_not_called()
            workflow.assert_not_called()


if __name__ == "__main__":
    unittest.main()
