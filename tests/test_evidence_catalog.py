import copy
import json
import tempfile
import unittest
from pathlib import Path

from evidence_catalog import build_catalog, materialize_plan, contains_quote, selected_name_supported
from resume_validation import candidate_source_text
from tailoring import build_tailoring_plan, validate_plan, job_requirements
from tests.resume_fixture import LATEX


JOB = "Python SQL development and relevant computer science coursework"
COURSES = [
    ("CS577", "intro to Data Science and Principles"),
    ("CS250", "Software Principles"),
    ("CS210", "Data Structures"),
    ("CS349", "Computer Architecture"),
]
BASE = candidate_source_text(LATEX)
EVIDENCE = {
    f"course:{i}:chunk:0": f"Chunk ID: course:{i}:chunk:0\nType: course\nCourse code: {code}\nCourse: {name}\nCategory: Computer Science"
    for i, (code, name) in enumerate(COURSES, 1)
}
SOURCES = {"base_resume": BASE, **EVIDENCE}


def id_plan(catalog):
    eid = next(eid for eid, r in catalog["evidence"].items() if "Built Python and SQL" in r["text"])
    return {
        "requirements": [{"id": "R1", "support": "direct", "evidence_ids": [eid], "action": "Prioritize the documented Python and SQL work."}],
        "projects": {"include_ids": list(catalog["projects"]), "reason": "API development is relevant."},
        "courses": {"include_ids": list(catalog["courses"]), "reason": "Coursework complements the small API project."},
    }


class EvidenceCatalogTests(unittest.TestCase):
    def test_screenshot_course_labels_resolve_against_separate_record_fields(self):
        for code, name in COURSES:
            with self.subTest(code=code):
                self.assertTrue(selected_name_supported(f"{code}: {name}", "courses", SOURCES))
                self.assertTrue(selected_name_supported(f"{code} ({name})", "courses", SOURCES))
                self.assertTrue(selected_name_supported(code, "courses", SOURCES))

    def test_course_code_and_name_must_belong_to_same_record(self):
        for name in ("CS577: Software Principles", "CS999: Data Structures", "CS210: Distributed Systems", "CS210: Data"):
            self.assertFalse(selected_name_supported(name, "courses", SOURCES), name)

    def test_tex_courses_and_nested_project_titles_are_catalogued(self):
        source = BASE.replace("{Python API}{2026}", r"{\textbf{Python API} $|$ \emph{Python, SQL}}{2026}")
        source += r"\section{Relevant Courses}\textbf{CS}{: CS577 (intro to Data Science and Principles), CS349 (Computer Architecture)}"
        catalog = build_catalog({"base_resume": source})
        self.assertEqual([r["name"] for r in catalog["projects"].values()], ["Python API"])
        self.assertEqual([r["code"] for r in catalog["courses"].values()], ["CS577", "CS349"])

    def test_legacy_quotes_allow_display_markup_but_reject_changed_facts(self):
        source = r"\resumeItem{Built \textbf{Python} and SQL services; increased bookings by 18\%.}"
        self.assertTrue(contains_quote("Built Python and SQL services; increased bookings by 18%.", source))
        self.assertTrue(contains_quote("Created O'Quinn's API in 2025-2027.", "Created O’Quinn’s API in 2025–2027."))
        for quote in ("Built Go and SQL services; increased bookings by 18%.",
                      "Built Python and SQL services; increased bookings by 81%.",
                      "Built Python and SQL services; increased bookings by 18 hours."):
            self.assertFalse(contains_quote(quote, source))
        self.assertFalse(contains_quote("Deployed services on AWS", "Did not deploy services on AWS"))

    def test_materialization_copies_real_quotes_and_course_pairs(self):
        catalog = build_catalog(SOURCES)
        raw = id_plan(catalog)
        # Ignore fabricated auxiliary labels/quotes when a catalog ID is selected.
        raw["requirements"][0]["evidence"] = [{"source_id": "job_posting", "quote": "Invented Go achievement"}]
        raw["courses"]["include"] = ["CS999: Invented course"]
        plan = materialize_plan(raw, catalog)
        self.assertEqual(validate_plan(plan, job_requirements(JOB), SOURCES), [])
        citation = plan["requirements"][0]["evidence"][0]
        self.assertIn(citation["quote"], SOURCES[citation["source_id"]])
        self.assertNotIn("Invented", json.dumps(plan))
        self.assertEqual(plan["courses"]["include"], [f"{code}: {name}" for code, name in COURSES])

    def test_unknown_or_wrong_kind_ids_are_rejected(self):
        catalog = build_catalog(SOURCES)
        for section, field, invalid in (("requirements", "evidence_ids", ["E999"]),
                                        ("courses", "include_ids", ["P001"]),
                                        ("projects", "include_ids", ["P999"])):
            plan = id_plan(catalog)
            target = plan[section][0] if section == "requirements" else plan[section]
            target[field] = invalid
            with self.assertRaises(ValueError):
                materialize_plan(plan, catalog)

    def test_id_plan_completes_without_transcribing_any_source_quotes(self):
        class Model:
            def invoke(self, messages):
                payload = json.loads(messages[1]["content"])
                catalog = payload["evidence_catalog"]
                assert all("quote" not in r for r in catalog["evidence"].values())
                assert "evidence_ids" in messages[0]["content"]
                return type("Response", (), {"text": json.dumps(id_plan(catalog))})()
        with tempfile.TemporaryDirectory() as directory:
            plan = build_tailoring_plan(Model(), JOB, BASE, EVIDENCE, job_requirements(JOB), directory)
            self.assertTrue((Path(directory) / "evidence_catalog.json").is_file())
            self.assertEqual(json.loads((Path(directory) / "plan_validation_1.json").read_text()), {"issues": []})
            self.assertFalse((Path(directory) / "plan_attempt_2.md").exists())
            self.assertEqual(validate_plan(plan, job_requirements(JOB), SOURCES), [])

    def test_legacy_plan_from_screenshot_pattern_can_pass_without_weakening_pairs(self):
        plan = {
            "requirements": [{"id": "R1", "support": "direct", "action": "Prioritize backend work.",
                              "evidence": [{"source_id": "base_resume", "quote": "Built Python and SQL services; increased bookings by 18%."}]}],
            "projects": {"include": ["Python API"], "reason": "Relevant API work."},
            "courses": {"include": [f"{c}: {n}" for c, n in COURSES], "reason": "Coursework adds academic evidence."},
        }
        self.assertEqual(validate_plan(plan, job_requirements(JOB), SOURCES), [])
        bad = copy.deepcopy(plan)
        bad["courses"]["include"][0] = "CS577: Software Principles"
        self.assertTrue(validate_plan(bad, job_requirements(JOB), SOURCES))
