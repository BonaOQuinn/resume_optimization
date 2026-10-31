import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from backend.prepare_documents import (
    prepare_all_documents,
    prepare_career_documents,
    prepare_course_documents,
    prepare_documents,
    prepare_skill_documents,
)

ROOT = Path(__file__).resolve().parents[1]


class DocumentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.database = self.folder / "resume.db"
        self.connection = sqlite3.connect(self.database)
        self.addCleanup(self.connection.close)
        self.connection.executescript((ROOT / "backend/schema.sql").read_text())
        self.connection.executemany(
            "INSERT INTO career_items VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(1, "experience", "Engineer", "O'Quinn & Co.", None, "experience.md", 0),
             (2, "project", "Vision project", None, "in progress", "projects.md", 1),
             (3, "project", "Empty project", None, None, "projects.md", 0)],
        )
        self.first = "I built an API with C++ and Node.js for the team’s users."
        self.second = "It handles the requests without removing any of the context.\nAnother sentence."
        self.connection.executemany(
            "INSERT INTO bullets VALUES (?, ?, ?, ?)",
            [(1, 1, self.second, 1), (2, 1, self.first, 0), (3, 2, "Tracked a player.", 0)],
        )
        self.connection.executemany(
            "INSERT INTO courses VALUES (?, ?, ?, ?, ?)",
            [(1, "CS549", "Machine Learning", "AI", "relevant_courses.md"),
             (2, None, "A course with no code", "Other", "relevant_courses.md")],
        )
        self.connection.execute("INSERT INTO skills VALUES (1, ?, ?, ?)",
                                ("C++ / Node.js", "Languages", "tech_skills.md"))
        self.connection.commit()

    def test_career_documents_preserve_complete_text_and_order(self):
        documents = prepare_career_documents(self.connection)
        self.assertEqual([d["id"] for d in documents],
                         ["career_item:1", "career_item:3", "career_item:2"])
        self.assertEqual(documents[0]["text"],
                         "Type: experience\nTitle: Engineer\nOrganization: O'Quinn & Co."
                         f"\n\nAccomplishments:\n- {self.first}\n- {self.second}")
        self.assertEqual(documents[0]["metadata"]["source_file"], "experience.md")
        self.assertEqual(documents[0]["metadata"]["record_id"], 1)
        self.assertEqual(documents[1]["text"], "Type: project\nTitle: Empty project")
        self.assertIn("Status: in progress", documents[2]["text"])
        self.assertNotIn("Organization:", documents[2]["text"])

    def test_filtering_and_caller_connection(self):
        self.connection.row_factory = sqlite3.Row
        self.assertEqual(len(prepare_career_documents(self.connection, "experience")), 1)
        self.assertEqual(len(prepare_career_documents(self.connection, "project")), 2)
        with self.assertRaises(ValueError):
            prepare_career_documents(self.connection, "other")
        self.assertIs(self.connection.row_factory, sqlite3.Row)
        self.assertEqual(self.connection.execute("SELECT 1").fetchone()[0], 1)

    def test_courses_and_skills_preserve_names_and_categories(self):
        courses = prepare_course_documents(self.connection)
        self.assertEqual(courses[0]["text"],
                         "Type: course\nCourse code: CS549\nCourse: Machine Learning\nCategory: AI")
        self.assertNotIn("Course code:", courses[1]["text"])
        self.assertIsNone(courses[1]["metadata"]["code"])
        skills = prepare_skill_documents(self.connection)
        self.assertEqual(skills[0]["text"], "Type: skill\nSkill: C++ / Node.js\nCategory: Languages")
        self.assertEqual(skills[0]["metadata"]["source_table"], "skills")

    def test_aggregate_ids_are_distinct_repeatable_and_read_only(self):
        before = self.database.read_bytes()
        documents = prepare_documents(self.database)
        self.assertEqual(len(documents), 6)
        self.assertEqual(len({d["id"] for d in documents}), 6)
        self.assertEqual(documents, prepare_all_documents(self.connection))
        self.assertEqual(documents, prepare_documents(self.database))
        self.assertEqual(self.database.read_bytes(), before)

    def test_empty_tables_return_no_documents(self):
        with closing(sqlite3.connect(":memory:")) as connection:
            connection.executescript((ROOT / "backend/schema.sql").read_text())
            self.assertEqual(prepare_all_documents(connection), [])

    def cli(self, *extra):
        return subprocess.run(
            [sys.executable, str(ROOT / "backend/prepare_documents.py"),
             "--database", str(self.database), *extra],
            cwd=self.folder, capture_output=True, text=True, encoding="utf-8")

    def test_cli_json_and_output_from_another_directory(self):
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        documents = json.loads(result.stdout)
        self.assertEqual(documents, prepare_documents(self.database))
        self.assertIn("team’s", result.stdout)
        output = self.folder / "nested/documents.json"
        result = self.cli("--output", str(output))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), documents)

    def test_missing_database_and_protected_output(self):
        missing = self.folder / "missing.db"
        with self.assertRaises(sqlite3.OperationalError):
            prepare_documents(missing)
        self.assertFalse(missing.exists())
        before = self.database.read_bytes()
        result = self.cli("--output", str(self.database))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output must be different", result.stderr)
        self.assertEqual(self.database.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
