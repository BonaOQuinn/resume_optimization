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
        with patch("resume_validation.find_latex_compiler", return_value=None):
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

    def test_body_only_response_is_wrapped_in_the_approved_preamble(self):
        from tests.resume_fixture import BODY
        import re
        assembled = extract_complete_latex("```latex\n" + BODY.strip() + "\n```")
        self.assertEqual(re.sub(r"\s+", "", assembled), re.sub(r"\s+", "", LATEX))
        self.assertEqual(validate_resume(extract_complete_latex(BODY), [SOURCE]), [])
        # Prose, or text without any resume section, is not a body.
        self.assertIsNone(extract_complete_latex("Here is your resume: " + BODY))
        self.assertIsNone(extract_complete_latex(r"\textbf{Only a name}"))

    def test_mangled_model_preamble_is_replaced_not_rejected(self):
        # The furthest real run failed here: a local model collapsed the macro
        # definitions and dropped row breaks from the template preamble.
        mangled = LATEX.replace("\\newcommand{\\resumeItem}[1]{\n  \\item\\small{\n    {#1 \\vspace{-2pt}}\n  }\n}",
                                "\\newcommand{\\resumeItem}[1]{\\item\\small{#1}}").replace("{article}", "{moderncv}")
        self.assertIn(r"\item\small{#1}}", mangled)
        latex = extract_complete_latex(mangled)
        self.assertEqual(latex, LATEX + "\n")
        self.assertEqual(validate_resume(latex, [SOURCE]), [])

    def test_missing_latex_package_stops_retries_and_keeps_reviewed_draft(self):
        from resume_validation import MISSING_PACKAGE_PREFIX, validate_and_export
        missing = ([MISSING_PACKAGE_PREFIX + "fullpage.sty. Install it."], None, "FAIL")
        model = RecordingChatModel(responses=[AIMessage(content=PASS)])
        calls = []
        def generate(issues):
            calls.append(issues)
            return {"response": LATEX}
        with tempfile.TemporaryDirectory() as directory, patch("resume_validation.compile_latex", return_value=missing):
            saved = validate_and_export(Path(directory) / "run", generate, model, SOURCE, require_pdf=True)
            report = json.loads(saved["validation"].read_text())
            self.assertEqual(len(calls), 1)  # No futile regeneration.
            self.assertEqual(report["status"], "FAIL")
            self.assertIn("fullpage.sty", report["error"])
            self.assertNotIn("latex", saved)
            self.assertEqual(saved["unverified_latex"].read_text(encoding="utf-8"), LATEX + "\n")

    def test_markup_only_correction_skips_repeated_model_reviews(self):
        from resume_validation import validate_and_export
        failed = (["LaTeX compilation failed: bad spacing"], None, "FAIL")
        passed = ([], b"%PDF-1.5", "PASS")
        model = RecordingChatModel(responses=[AIMessage(content=PASS)])  # Only one review available.
        drafts = iter([LATEX, LATEX.replace("\\section{Projects}", "\\section{Projects}\n\\vspace{-2pt}")])
        with tempfile.TemporaryDirectory() as directory, patch("resume_validation.compile_latex", side_effect=[failed, passed]):
            saved = validate_and_export(Path(directory) / "run", lambda _: {"response": next(drafts)}, model, SOURCE)
            report = json.loads(saved["validation"].read_text())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(model.seen), 1)
        self.assertEqual(report["attempts"][1]["grounding"], "PASS (unchanged text)")

    def test_reviewers_receive_the_body_not_the_fixed_preamble(self):
        model = RecordingChatModel(responses=[AIMessage(content=PASS)])
        grounding_review(model, LATEX, [SOURCE])
        payload = json.loads(model.seen[0][1].content)
        self.assertNotIn("documentclass", payload["candidate_latex"])
        self.assertIn("Demo Candidate", payload["candidate_latex"])

    def test_reviewers_do_not_receive_a_dangling_end_document(self):
        # A real run failed twice because the tailoring reviewer saw a body that
        # ended with \end{document} and reported the document as broken.
        from tailoring import review_tailoring
        plan = {"requirements": [], "projects": {"include": [], "reason": "n/a"}, "courses": {"include": [], "reason": "n/a"}}
        model = RecordingChatModel(responses=[AIMessage(content=PASS), AIMessage(content=PASS)])
        grounding_review(model, LATEX, [SOURCE])
        review_tailoring(model, LATEX, "Python", plan, [SOURCE])
        for system, user in model.seen:
            body = json.loads(user.content)["candidate_latex"]
            self.assertNotIn(r"\end{document}", body)
            self.assertNotIn(r"\begin{document}", body)
            self.assertIn("Demo Candidate", body)
            self.assertIn("only the resume body", system.content)

    def test_writer_notes_after_the_body_stay_out_of_the_resume(self):
        from tests.resume_fixture import BODY
        from resume_validation import NOTES_MARKER, split_writer_notes
        body = BODY.replace(r"\textbf{Languages}: Python, SQL", "\\begin{itemize}\n\\item Python, SQL\n\\end{itemize}").strip()
        questions = "1. **Project end date:** the source says \"Dev 2025\". Is it December?"
        clean = extract_complete_latex(body)
        for response in (body + "\n\n---\nItems to confirm with the candidate (not part of the resume):\n" + questions,
                         body + "\n" + NOTES_MARKER + "\n" + questions):
            with self.subTest(response=response[-120:]):
                self.assertEqual(extract_complete_latex(response), clean)
                self.assertEqual(validate_resume(extract_complete_latex(response), [SOURCE]), [])
                self.assertIn("Dev 2025", split_writer_notes(response)[1])
        self.assertEqual(split_writer_notes(body), (body, ""))

    def test_writer_notes_are_saved_for_the_user_when_a_resume_is_exported(self):
        from tests.resume_fixture import BODY
        from resume_validation import NOTES_MARKER
        body = BODY.replace(r"\textbf{Languages}: Python, SQL", "\\begin{itemize}\n\\item Python, SQL\n\\end{itemize}").strip()
        response = body + "\n" + NOTES_MARKER + "\nIs the project end month December?"
        model = RecordingChatModel(responses=[AIMessage(content=response), AIMessage(content=PASS)])
        with tempfile.TemporaryDirectory() as directory:
            saved = run_direct_resume_optimization("Python", SOURCE, Path(directory) / "run", model=model)
            self.assertEqual(json.loads(saved["validation"].read_text())["status"], "PASS")
            self.assertNotIn("December", saved["latex"].read_text(encoding="utf-8"))
            self.assertEqual(saved["notes"].read_text(encoding="utf-8").strip(), "Is the project end month December?")

    def test_contact_urls_must_match_not_just_prefix_a_source_url(self):
        source = SOURCE + "\nhttps://example.org/demo-profile"
        candidate = LATEX.replace("Python, SQL", r"\url{https://example.org}")
        self.assertTrue(validate_resume(candidate, [source]))


if __name__ == "__main__":
    unittest.main()
