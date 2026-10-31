import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from backend.prepare_job_document import prepare_job_document

ROOT = Path(__file__).resolve().parents[1]


class JobDocumentTests(unittest.TestCase):
    def test_full_text_preserved_and_sections_distinct(self):
        description = "Build the team's C++ / Node.js services.\n\nKeep all of the context for our users."
        qualifications = "- Work with Python and SQL.\n- Explain the model’s behavior."
        preferred = "Experience with AWS is a plus."
        document = prepare_job_document("Engineer", description, qualifications, preferred)
        self.assertEqual(document["text"],
                         f"Job title: Engineer\n\nDescription:\n{description}"
                         f"\n\nRequired qualifications:\n{qualifications}"
                         f"\n\nPreferred qualifications:\n{preferred}")
        self.assertEqual(document["metadata"], {"item_type": "job", "title": "Engineer"})

    def test_optional_fields_trim_edges_without_changing_internal_spaces(self):
        document = prepare_job_document("  Engineer \n", "\nBuild  the services.\n", " \n", "")
        self.assertEqual(document["text"], "Job title: Engineer\n\nDescription:\nBuild  the services.")
        document = prepare_job_document("Engineer", "Build services.", preferred_qualifications="AWS")
        self.assertIn("Preferred qualifications:\nAWS", document["text"])
        self.assertNotIn("Required qualifications:", document["text"])

    def test_required_fields_and_invalid_types(self):
        for title, description in (("", "Text"), ("Engineer", " \n")):
            with self.assertRaises(ValueError):
                prepare_job_document(title, description)
        for fields in ((None, "Text", "", ""), ("Engineer", [], "", ""),
                       ("Engineer", "Text", None, ""), ("Engineer", "Text", "", 1)):
            with self.assertRaises(TypeError):
                prepare_job_document(*fields)

    def test_deterministic_id_changes_when_content_or_section_changes(self):
        first = prepare_job_document("Engineer", "Text", "Python")
        self.assertEqual(first, prepare_job_document("Engineer", "Text", "Python"))
        self.assertEqual(first["id"], prepare_job_document(" Engineer ", "Text", "Python")["id"])
        self.assertNotEqual(first["id"], prepare_job_document("Engineer", "Changed", "Python")["id"])
        self.assertNotEqual(first["id"], prepare_job_document("Engineer", "Text", "", "Python")["id"])
        self.assertTrue(first["id"].startswith("job:"))

    def run_cli(self, pasted, *args):
        return subprocess.run([sys.executable, str(ROOT / "backend/prepare_job_document.py"), *args],
                              input=pasted, capture_output=True, text=True, encoding="utf-8")

    def test_pasted_multiline_fields_and_json_output(self):
        pasted = "Engineer\nBuild services.\n\nFor the team’s users.\n.done\nPython\n.done\n.done\n"
        result = self.run_cli(pasted)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), prepare_job_document(
            "Engineer", "Build services.\n\nFor the team’s users.", "Python"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested/job.json"
            result = self.run_cli(pasted, "--output", str(output))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), json.loads(
                self.run_cli(pasted).stdout))

    def test_incomplete_paste_does_not_write_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "job.json"
            result = self.run_cli("Engineer\nIncomplete description\n", "--output", str(output))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("before .done", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
