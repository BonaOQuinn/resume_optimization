"""Exercise SQLite-to-Chroma indexing without network calls or API credentials."""

import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import Embeddings

import main as indexing


ROOT = Path(__file__).resolve().parents[1]


class OfflineEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0, 0.5] for text in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class IndexingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.database = self.folder / "resume.db"
        self.persist_directory = self.folder / "chroma"
        with sqlite3.connect(self.database) as connection:
            connection.executescript((ROOT / "backend/schema.sql").read_text())

    def test_documents_persist_and_reindex_updates_existing_ids(self):
        with sqlite3.connect(self.database) as connection:
            connection.executemany(
                "INSERT INTO career_items VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(1, "experience", "Engineer", None, None, "experience.md", 0),
                 (2, "project", "Vision project", None, None, "projects.md", 1)],
            )
            connection.execute(
                "INSERT INTO bullets VALUES (1, 1, ?, 0)",
                ("Built Python services for the team's users.",),
            )
            connection.execute(
                "INSERT INTO courses VALUES (1, 'CS101', 'Computing', 'CS', 'courses.md')"
            )
            connection.execute(
                "INSERT INTO skills VALUES (1, 'Python', 'Languages', 'skills.md')"
            )

        # Only the external embedding provider is replaced; Chroma writes to disk.
        with patch.object(indexing, "OpenAIEmbeddings", return_value=OfflineEmbeddings()):
            self.assertEqual(
                indexing.index_resume_documents(self.database, self.persist_directory), 4
            )
            with sqlite3.connect(self.database) as connection:
                connection.execute("UPDATE career_items SET title='Senior Engineer' WHERE id=1")
            self.assertEqual(
                indexing.index_resume_documents(self.database, self.persist_directory), 4
            )
            reopened = indexing.create_vector_store(self.persist_directory)
            saved = reopened.get(include=["documents", "metadatas", "embeddings"])
            self.assertEqual(set(saved["ids"]),
                             {"career_item:1", "career_item:2", "course:1", "skill:1"})
            self.assertEqual(len(saved["ids"]), 4)
            self.assertEqual(len(saved["embeddings"]), 4)
            self.assertTrue((self.persist_directory / "chroma.sqlite3").is_file())
            position = saved["ids"].index("career_item:1")
            self.assertIn("Title: Senior Engineer", saved["documents"][position])
            self.assertNotIn("organization", saved["metadatas"][position])
            self.assertEqual(len(reopened.similarity_search("Python", k=1)), 1)

    def test_empty_and_missing_database_do_not_create_a_store(self):
        with patch.object(indexing, "create_vector_store") as create_store:
            self.assertEqual(
                indexing.index_resume_documents(self.database, self.persist_directory), 0
            )
            missing = self.folder / "missing.db"
            with self.assertRaisesRegex(FileNotFoundError, "Create and populate"):
                indexing.index_resume_documents(missing, self.persist_directory)
            create_store.assert_not_called()
        self.assertFalse(missing.exists())
        self.assertFalse(self.persist_directory.exists())

    def test_cli_help_does_not_need_an_api_key(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "--help"],
            cwd=self.folder,
            env={**os.environ, "OPENAI_API_KEY": ""},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--persist-directory", result.stdout)


if __name__ == "__main__":
    unittest.main()
