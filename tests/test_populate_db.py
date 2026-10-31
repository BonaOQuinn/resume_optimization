"""Exercise imports, updates, and failures against temporary SQLite databases."""

import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from backend.populate_db import SOURCE_FILES, SourceError, load_sources, populate_database

ROOT = Path(__file__).resolve().parents[1]


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.sources = self.folder / "sources"
        self.sources.mkdir()
        for name in SOURCE_FILES:
            shutil.copyfile(ROOT / name, self.sources / name)
        self.database = self.folder / "db" / "resume.db"

    def rows(self, sql, parameters=()):
        with closing(sqlite3.connect(self.database)) as connection:
            return connection.execute(sql, parameters).fetchall()

    def snapshot(self):
        with closing(sqlite3.connect(self.database)) as connection:
            return list(connection.iterdump())

    def test_actual_sources_and_rerun(self):
        data = load_sources(self.sources)
        result = populate_database(self.database, data)
        self.assertEqual(result, dict(experience=3, projects=4, bullets=22, courses=12, skills=33))
        self.assertEqual(self.rows("SELECT COUNT(*) FROM career_items"), [(7,)])
        self.assertEqual(self.rows("SELECT COUNT(*) FROM bullets"), [(22,)])
        self.assertEqual(self.rows("PRAGMA foreign_key_check"), [])
        self.assertEqual(self.rows("SELECT organization, status FROM career_items WHERE title='Handles'"),
                         [("personal", "in progress")])
        self.assertEqual(self.rows("SELECT category FROM courses WHERE code='CS549'"), [("AI",)])
        bullets = self.rows("SELECT text FROM bullets ORDER BY id")
        self.assertTrue(any("backend services and user-facing features." in text for text, in bullets))
        self.assertTrue(any("35maintain" in text for text, in bullets))
        self.assertIn(("linux / openbsd / azure /",), self.rows("SELECT name FROM skills"))
        original = self.snapshot()
        populate_database(self.database, data)
        self.assertEqual(self.snapshot(), original)

    def test_changes_removals_and_unrelated_records(self):
        data = load_sources(self.sources)
        populate_database(self.database, data)
        original_id = self.rows("SELECT id FROM career_items WHERE title='Handles'")[0][0]
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("INSERT INTO skills(name, category, source_file) VALUES (?, ?, ?)",
                               ("Other record", "Other", "manual"))
            connection.commit()
        data["experience"].pop()
        data["projects"][2]["bullets"][0] = "Updated project's bullet with apostrophe."
        data["projects"][2]["bullets"].pop()
        data["courses"][0]["name"] = "Updated course"
        data["skills"].pop()
        data["skills"].append(dict(name="New skill", category="Tools", source_file="tech_skills.md"))
        populate_database(self.database, data)
        self.assertEqual(self.rows("SELECT id FROM career_items WHERE title='Handles'"), [(original_id,)])
        self.assertEqual(self.rows("SELECT text FROM bullets WHERE career_item_id=? ORDER BY sort_order",
                                  (original_id,))[0], ("Updated project's bullet with apostrophe.",))
        self.assertEqual(self.rows("SELECT COUNT(*) FROM career_items"), [(6,)])
        self.assertEqual(self.rows("SELECT COUNT(*) FROM bullets WHERE career_item_id=?", (original_id,)), [(2,)])
        self.assertEqual(self.rows("SELECT name FROM courses WHERE code='CS577'"), [("Updated course",)])
        self.assertIn(("Other record",), self.rows("SELECT name FROM skills"))
        self.assertIn(("New skill",), self.rows("SELECT name FROM skills"))
        self.assertEqual(self.rows("PRAGMA foreign_key_check"), [])

    def test_write_failure_rolls_back_every_section(self):
        data = load_sources(self.sources)
        populate_database(self.database, data)
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("""CREATE TRIGGER fail_skills BEFORE INSERT ON skills
                                  BEGIN SELECT RAISE(ABORT, 'blocked'); END""")
            connection.commit()
        original = self.snapshot()
        data["experience"].pop()
        data["projects"][0]["bullets"][0] = "Should be rolled back"
        data["skills"].append(dict(name="New skill", category="Tools", source_file="tech_skills.md"))
        with self.assertRaises(sqlite3.IntegrityError):
            populate_database(self.database, data)
        self.assertEqual(self.snapshot(), original)

    def test_bad_or_missing_source_does_not_create_database(self):
        for text in ("# Courses\n# AI\n- Missing separator", "# Courses\n# AI\n",
                     "# Courses\n# AI\n- CS1: A\n- CS1: B"):
            (self.sources / "relevant_courses.md").write_text(text, encoding="utf-8")
            result = self.run_cli()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Import failed:", result.stderr)
            self.assertFalse(self.database.exists())
        (self.sources / "experience.md").unlink()
        self.assertNotEqual(self.run_cli().returncode, 0)
        self.assertFalse(self.database.exists())

    def test_modern_headings_and_duplicate_career_entries(self):
        path = self.sources / "experience.md"
        path.write_text("# Experience\n## Engineer - Company\n- First line\ncontinued line\n", encoding="utf-8")
        data = load_sources(self.sources)
        self.assertEqual(data["experience"][0]["bullets"], ["First line continued line"])
        with path.open("a", encoding="utf-8") as stream:
            stream.write("## Engineer - Company\n- Another bullet\n")
        with self.assertRaises(SourceError):
            load_sources(self.sources)

    def run_cli(self, *extra):
        return subprocess.run([sys.executable, str(ROOT / "backend" / "populate_db.py"),
                               "--source-dir", str(self.sources), "--database", str(self.database),
                               *extra], cwd=self.folder, capture_output=True, text=True)

    def test_cli_from_another_directory_and_dry_run(self):
        result = self.run_cli("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("3 experience", result.stdout)
        self.assertFalse(self.database.exists())
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.database.exists())
        result = subprocess.run([sys.executable, "-m", "backend.populate_db", "--dry-run"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
