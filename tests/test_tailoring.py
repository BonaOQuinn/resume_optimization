import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from resume_pipeline import retrieve_evidence, run_validated_pipeline
from resume_validation import validate_resume, validate_and_export
from tailoring import job_requirements, validate_plan, build_tailoring_plan
from tests.resume_fixture import SOURCE, LATEX, PASS
from tests.test_resume_workflow import RecordingChatModel


def plan_for(job="Required: Python SQL backend engineering.", source=SOURCE):
    return {
        "requirements": [{"id": r["id"], "support": "direct", "evidence": [
            {"source_id": "base_resume", "quote": "Built Python and SQL services; increased bookings by 18%."}],
            "action": "Prioritize Python/SQL services and the documented bookings result."} for r in job_requirements(job)],
        "projects": {"include": ["Python API"], "reason": "Direct backend/API evidence."},
        "courses": {"include": [], "reason": "The API project directly supports the job."},
    }


class FakeStore:
    def __init__(self):
        self.queries = []

    def get(self, **kwargs):
        return {"ids": ["career_item:1:chunk:0"]}

    def similarity_search(self, query, **kwargs):
        self.queries.append((query, kwargs))
        category = kwargs["filter"]["item_type"]
        return [Document(id=f"{category}:1", page_content=SOURCE, metadata={"item_type": category})]


class TailoringTests(unittest.TestCase):
    def test_selected_course_record_reaches_writer_with_noncontiguous_label(self):
        from direct_resume import run_direct_resume_optimization
        evidence = "Chunk ID: course:1:chunk:0\nType: course\nCourse code: CS577\nCourse: Data Science\nCategory: AI"
        plan = plan_for()
        plan["courses"]["include"] = ["CS577: Data Science"]
        model = RecordingChatModel(responses=[AIMessage(content=LATEX), AIMessage(content=PASS), AIMessage(content=PASS)])
        with tempfile.TemporaryDirectory() as directory:
            run_direct_resume_optimization("Python SQL", SOURCE, Path(directory) / "run", model=model,
                                           evidence_documents=[evidence], tailoring_plan=plan, compile_pdf=False)
        payload = json.loads(model.seen[0][1].content.split("\n", 1)[1])
        self.assertEqual(payload["retrieved_candidate_evidence"], [evidence])

    def test_writer_and_reviewer_get_a_plan_without_duplicated_quotes(self):
        from direct_resume import run_direct_resume_optimization
        from tailoring import slim_plan
        plan = plan_for()
        slim = slim_plan(plan)
        self.assertNotIn("quote", json.dumps(slim))
        self.assertEqual([m["id"] for m in slim["requirements"]], [m["id"] for m in plan["requirements"]])
        self.assertEqual(slim["projects"]["include"], ["Python API"])
        improved = LATEX.replace("Built Python and SQL services;", "Developed Python services backed by SQL;")
        model = RecordingChatModel(responses=[AIMessage(content=improved), AIMessage(content=PASS), AIMessage(content=PASS)])
        with tempfile.TemporaryDirectory() as directory:
            run_direct_resume_optimization("Python SQL", SOURCE, Path(directory) / "run", model=model,
                                           tailoring_plan=plan, compile_pdf=False)
        for messages in (model.seen[0], model.seen[2]):  # Writer and tailoring reviewer.
            self.assertNotIn('"quote"', messages[1].content)

    def test_rare_late_requirement_is_retained_beside_frequent_generic_matches(self):
        class RankedStore:
            def similarity_search(self, query, **kwargs):
                category = kwargs["filter"]["item_type"]
                key = query if category == "skill" else "generic"
                return [Document(id=category + key, page_content=key, metadata={"item_type": category})]
        job = "\n".join([f"Skill{i}" for i in range(8)] * 3 + ["ZZZRareLateSkill"])
        evidence = retrieve_evidence(job, RankedStore())
        self.assertTrue(any("ZZZRareLateSkill" in e for e in evidence))

    def test_short_skills_and_late_requirements_are_searched(self):
        job = "# Qualifications\n" + "\n".join(f"- Requirement {i}" for i in range(12)) + "\n- SQL\n- Kubernetes"
        store = FakeStore()
        evidence = retrieve_evidence(job, store)
        self.assertEqual(len(job_requirements(job)), 14)
        self.assertIn("SQL", [q for q, _ in store.queries])
        self.assertIn("Kubernetes", [q for q, _ in store.queries])
        self.assertEqual({x[1]["filter"]["item_type"] for x in store.queries}, {"experience", "project", "skill", "course"})
        self.assertEqual(len(evidence), 4)

    def test_source_ids_coverage_and_exact_quotes_are_required(self):
        req = job_requirements("Required: Python SQL backend engineering.")
        plan = plan_for()
        self.assertEqual(validate_plan(plan, req, {"base_resume": SOURCE}), [])
        for mutate in (
            lambda p: p["requirements"].clear(),
            lambda p: p["requirements"][0]["evidence"][0].update(source_id="job_posting"),
            lambda p: p["requirements"][0]["evidence"][0].update(quote="Built Go financial microservices."),
            lambda p: p["requirements"][0].update(evidence=[]),
            lambda p: p["projects"].update(include=["Invented Finance Engine"]),
        ):
            broken = copy.deepcopy(plan)
            mutate(broken)
            self.assertTrue(validate_plan(broken, req, {"base_resume": SOURCE}))

    def test_unsupported_job_qualifications_can_be_gaps(self):
        job = "Go microservices and SOX compliance"
        plan = plan_for(job)
        plan["requirements"][0].update(support="gap", evidence=[], action="No Go or SOX evidence; do not claim either.")
        self.assertEqual(validate_plan(plan, job_requirements(job), {"base_resume": SOURCE}), [])

    def test_invalid_plans_are_repaired_before_drafting(self):
        good = plan_for()
        bad = copy.deepcopy(good)
        bad["requirements"][0]["evidence"][0]["source_id"] = "job_posting"
        model = RecordingChatModel(responses=[AIMessage(content=json.dumps(bad)), AIMessage(content=json.dumps(good))])
        with tempfile.TemporaryDirectory() as directory:
            plan = build_tailoring_plan(model, "Required: Python SQL backend engineering.", SOURCE, {}, job_requirements("Required: Python SQL backend engineering."), directory)
        self.assertEqual(plan["requirements"][0]["evidence"][0]["source_id"], "base_resume")
        self.assertEqual(len(model.seen), 2)

    def test_unrelated_resume_fails_tailoring_review_and_is_corrected(self):
        revision = '{"verdict":"REVISE","issues":["Prioritize the Python API project and clarify the SQL bullet."]}'
        improved = LATEX.replace("Built Python and SQL services;", "Developed Python services backed by SQL;")
        model = RecordingChatModel(responses=[
            AIMessage(content=json.dumps(plan_for())), AIMessage(content=LATEX), AIMessage(content=PASS),
            AIMessage(content=revision), AIMessage(content=improved), AIMessage(content=PASS), AIMessage(content=PASS),
        ])
        with tempfile.TemporaryDirectory() as directory:
            stages = []
            saved = run_validated_pipeline("Required: Python SQL backend engineering.", SOURCE, Path(directory) / "run", model=model, vector_store=FakeStore(), progress=lambda s, m: stages.append(s))
            report = json.loads(saved["validation"].read_text())
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["attempts"][0]["tailoring"], "REVISE")
            self.assertEqual(report["attempts"][1]["tailoring"], "PASS")
            self.assertIn("Developed Python", saved["latex"].read_text())
            self.assertTrue(saved["plan"].is_file())
            self.assertIn("planning", stages)
            self.assertIn("compiling", stages)
            self.assertEqual(stages[-1], "complete")

    def test_source_copy_and_bad_skill_row_breaks_are_rejected(self):
        bad = LATEX.replace("Python, SQL", "Python, SQL " + "\\" * 4)
        self.assertTrue(any("row break" in i for i in validate_resume(bad, [SOURCE])))
        model = RecordingChatModel(responses=[AIMessage(content=PASS)])
        with tempfile.TemporaryDirectory() as directory:
            saved = validate_and_export(Path(directory) / "run", lambda _: {"response": LATEX}, model, LATEX,
                                        job_document="Python", tailoring_plan=plan_for(), max_revisions=0)
            self.assertNotIn("latex", saved)
            self.assertIn("repeat the source", saved["validation"].read_text())
            self.assertEqual(model.seen, [])

    def test_missing_pdf_compiler_fails_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory, patch("resume_validation.find_latex_compiler", return_value=None), patch("agent.initialize_chat_model") as initialize:
            output = Path(directory) / "run"
            with self.assertRaisesRegex(ValueError, "pdflatex"):
                run_validated_pipeline("Python", SOURCE, output, require_pdf=True)
            initialize.assert_not_called()
            self.assertEqual(json.loads((output / "validation.json").read_text())["status"], "ERROR")

    def test_latex_only_skips_compiler_and_missing_packages_keep_a_log(self):
        from resume_validation import compile_latex
        model = RecordingChatModel(responses=[AIMessage(content=PASS)])
        with tempfile.TemporaryDirectory() as directory, patch("resume_validation.compile_latex") as compile_mock:
            saved = validate_and_export(Path(directory) / "run", lambda _: {"response": LATEX}, model, SOURCE, compile_pdf=False)
            self.assertIn("latex", saved)
            self.assertNotIn("pdf", saved)
            compile_mock.assert_not_called()
        import shutil
        if shutil.which("pdflatex"):
            with tempfile.TemporaryDirectory() as directory:
                log = Path(directory) / "compile.log"
                issues, pdf, status = compile_latex(LATEX.replace("Python, SQL", r"\undefinedMacro"), log_path=log)
                self.assertEqual(status, "FAIL")
                self.assertIsNone(pdf)
                self.assertIn("Undefined control sequence", log.read_text())

    def test_commented_template_examples_are_not_candidate_evidence(self):
        from resume_validation import candidate_source_text
        source = "% Template author Example Author\n" + LATEX.replace(r"\begin{document}", "\\begin{document}\n% Example GPA: 4.0")
        facts = candidate_source_text(source)
        self.assertNotIn("Example Author", facts)
        self.assertNotIn("GPA", facts)
        self.assertIn("Demo Candidate", facts)
        self.assertIn("2027", facts)

    def test_partial_artifacts_and_running_status_exist_during_review(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            class InterruptedModel:
                def invoke(self, messages):
                    self_status = json.loads((output / "validation.json").read_text())
                    assert self_status["status"] == "RUNNING"
                    assert self_status["stage"] == "grounding"
                    assert (output / "attempt_1.tex").exists()
                    raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                validate_and_export(output, lambda _: {"response": LATEX}, InterruptedModel(), SOURCE)
            self.assertEqual(json.loads((output / "validation.json").read_text())["status"], "INTERRUPTED")
            self.assertFalse((output / "optimized_resume.pdf").exists())
