import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage
from direct_resume import run_direct_resume_optimization
from resume_validation import validate_resume, compile_latex, grounding_review, extract_complete_latex
from tests.resume_fixture import LATEX, SOURCE, PASS
from tests.test_resume_workflow import RecordingChatModel


class ResumeValidationTests(unittest.TestCase):
    def test_approved_template_and_future_graduation_are_accepted(self):
        self.assertEqual(validate_resume(LATEX, [SOURCE]), [])

    def test_moderncv_margins_markdown_and_bad_escapes_are_rejected(self):
        cases = [
            LATEX.replace("{article}", "{moderncv}"),
            LATEX.replace("{-0.5in}", "{-0.1in}"),
            LATEX.replace("Python, SQL", "**Python**, SQL"),
            LATEX.replace("Python, SQL", "[Profile](https://example.org)"),
            LATEX.replace(r"18\%", r"18\\%"),
            LATEX.replace(r"18\%", "18%"),
            LATEX.replace("Python, SQL", "GPA: 3.8"),
            LATEX.replace(r"\section{Education}", r"\section{Summary}"),
        ]
        for candidate in cases:
            with self.subTest(candidate=candidate):
                self.assertTrue(validate_resume(candidate, [SOURCE]))

    def test_percentages_years_and_contacts_need_candidate_source_support(self):
        bad = LATEX.replace(r"18\%", r"35\%").replace("2027", "2028").replace("demo@candidate.test", "invented@candidate.test")
        issues = "\n".join(validate_resume(bad, [SOURCE + "\nReduced tokens by 35maintain efficiency"]))
        self.assertIn("35%", issues)
        self.assertIn("2028", issues)
        self.assertIn("invented@candidate.test", issues)

    def test_single_fenced_document_is_extracted_from_explanatory_prose(self):
        response = "Here is the draft:\n```latex\n" + LATEX + "\n```\nNotes outside the document."
        self.assertEqual(extract_complete_latex(response), LATEX + "\n")

    def test_correction_replaces_rejected_template_before_export(self):
        bad = LATEX.replace("{article}", "{moderncv}").replace(r"18\%", r"35\%")
        model = RecordingChatModel(responses=[AIMessage(content=bad), AIMessage(content=LATEX), AIMessage(content=PASS)])
        with tempfile.TemporaryDirectory() as directory:
            saved = run_direct_resume_optimization("Required: Python; improve results by 35%", SOURCE, Path(directory) / "run", model=model)
            self.assertEqual(saved["latex"].read_text(), LATEX + "\n")
            report = json.loads(saved["validation"].read_text())
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(len(report["attempts"]), 2)
            self.assertIn("Percentage 35%", "\n".join(report["attempts"][0]["issues"]))

    def test_grounding_reviewer_catches_date_reattribution(self):
        bad = LATEX.replace("2025--Present", "2026--Present")  # Both years exist in source.
        revision = '{"verdict":"REVISE","issues":["Engineer start year changed from 2025 to 2026."]}'
        model = RecordingChatModel(responses=[AIMessage(content=bad), AIMessage(content=revision), AIMessage(content=LATEX), AIMessage(content=PASS)])
        with tempfile.TemporaryDirectory() as directory:
            saved = run_direct_resume_optimization("Python", SOURCE, Path(directory) / "run", model=model)
            self.assertIn("2025--Present", saved["latex"].read_text())
            report = json.loads(saved["validation"].read_text())
            self.assertEqual(report["attempts"][0]["grounding"], "REVISE")

    def test_failed_review_or_repeated_invalid_outputs_never_export(self):
        model = RecordingChatModel(responses=[AIMessage(content=LATEX), AIMessage(content="Looks good!")])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            saved = run_direct_resume_optimization("Python", SOURCE, output, model=model)
            report = json.loads(saved["validation"].read_text())
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(len(report["attempts"]), 3)
            self.assertNotIn("latex", saved)
            self.assertFalse((output / "optimized_resume.pdf").exists())

    def test_compile_error_triggers_correction_before_pdf_export(self):
        import shutil
        if not shutil.which("pdflatex"):
            self.skipTest("pdflatex required")
        bad = LATEX.replace("Python, SQL", r"\undefinedMacro")
        model = RecordingChatModel(responses=[AIMessage(content=bad), AIMessage(content=PASS), AIMessage(content=LATEX), AIMessage(content=PASS)])
        with tempfile.TemporaryDirectory() as directory:
            saved = run_direct_resume_optimization("Python", SOURCE, Path(directory) / "run", model=model, require_pdf=True)
            report = json.loads(saved["validation"].read_text())
            self.assertEqual(report["attempts"][0]["compilation"], "FAIL")
            self.assertEqual(report["attempts"][1]["compilation"], "PASS")
            self.assertTrue(saved["pdf"].read_bytes().startswith(b"%PDF-"))

    def test_compiler_requirement_and_compile_failure_are_explicit(self):
        with patch("resume_validation.shutil.which", return_value=None):
            self.assertEqual(compile_latex(LATEX)[2], "NOT_RUN")
            self.assertEqual(compile_latex(LATEX, require_pdf=True)[2], "FAIL")
        # Use the actual compiler when installed; undefined macros must fail.
        import shutil
        if shutil.which("pdflatex"):
            self.assertEqual(compile_latex(LATEX.replace("Python, SQL", r"\undefinedMacro"))[2], "FAIL")

    def test_provider_failure_is_recorded_without_exporting_a_draft(self):
        from unittest.mock import Mock
        model = Mock()
        model.invoke.side_effect = ConnectionError("Test provider unavailable")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with self.assertRaises(ConnectionError):
                run_direct_resume_optimization("Python", SOURCE, output, model=model)
            report = json.loads((output / "validation.json").read_text())
            self.assertEqual(report["status"], "ERROR")
            self.assertFalse((output / "optimized_resume.tex").exists())

    def test_contact_urls_must_match_not_just_prefix_a_source_url(self):
        source = SOURCE + "\nhttps://example.org/demo-profile"
        candidate = LATEX.replace("Python, SQL", r"\url{https://example.org}")
        self.assertTrue(validate_resume(candidate, [source]))


if __name__ == "__main__":
    unittest.main()
