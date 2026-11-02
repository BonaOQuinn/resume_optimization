"""SQLite -> chunks -> persistent Chroma -> generation/review -> compiled PDF.

Default tests replace the embedding provider and chat responses, not Chroma,
retrieval, the agent graph, validators, file export, or the LaTeX compiler.
Set RESUME_LIVE_EMBEDDINGS=1 to additionally use downloaded MiniLM weights.
"""

import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage
import agent
import main as indexing
from backend.chunk_documents import split_resume_documents
from prompts import LATEX_PATH, EVIDENCE_PATH
from resume_pipeline import run_validated_pipeline
from run_resume_agent import run_resume_optimization
from tests.resume_fixture import LATEX, SOURCE, PASS
from tests.test_index_resume_documents import OfflineEmbeddings
from tests.test_resume_workflow import RecordingChatModel
from tests.tokenizer_fixture import make_tokenizer


@unittest.skipUnless(shutil.which("pdflatex"), "Integration tests require pdflatex")
class FullPipelineTests(unittest.TestCase):
    def run_pipeline(self, *, live_embeddings=False, agent_mode=False):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            database, persist = root / "resume.db", root / "chroma"
            with sqlite3.connect(database) as connection:
                connection.executescript((Path(indexing.__file__).parent / "backend/schema.sql").read_text())
                connection.execute("INSERT INTO career_items VALUES (1, 'experience', 'Engineer', 'Example Company', NULL, 'experience.md', 0)")
                connection.execute("INSERT INTO career_items VALUES (2, 'project', 'Python API', NULL, NULL, 'projects.md', 1)")
                connection.execute("INSERT INTO bullets VALUES (1, 1, 'Built Python and SQL services; increased bookings by 18%.', 0)")
                connection.execute("INSERT INTO bullets VALUES (2, 2, 'Built an API in Python.', 0)")
                connection.execute("INSERT INTO courses VALUES (1, 'CS101', 'Computing', 'CS', 'courses.md')")
            if not live_embeddings:
                tokenizer = make_tokenizer()
                stack.enter_context(patch.object(indexing, "split_resume_documents", side_effect=lambda docs: split_resume_documents(docs, tokenizer=tokenizer)))
                stack.enter_context(patch.object(indexing, "HuggingFaceEmbeddings", return_value=OfflineEmbeddings()))
            self.assertEqual(indexing.index_resume_documents(database, persist), 3)
            store = indexing.create_vector_store(persist)
            self.assertTrue((persist / "chroma.sqlite3").exists())
            if agent_mode:
                model = RecordingChatModel(responses=[
                    AIMessage(content="", tool_calls=[{"name": "search_resume", "args": {"query": "Python SQL services"}, "id": "retrieve"}]),
                    AIMessage(content="", tool_calls=[
                        {"name": "write_file", "args": {"file_path": LATEX_PATH, "content": LATEX}, "id": "draft"},
                        {"name": "write_file", "args": {"file_path": EVIDENCE_PATH, "content": "Supported Python and SQL services; source experience.md; career_item:1:chunk:0."}, "id": "evidence"},
                    ]), AIMessage(content="Draft written."), AIMessage(content=PASS),
                ])
                with patch.object(agent, "get_resume_vector_store", return_value=store):
                    saved = run_resume_optimization("Required: Python SQL backend engineering.", SOURCE, root / "run", model=model, require_pdf=True)
            else:
                bad = LATEX.replace("{article}", "{moderncv}").replace(r"18\%", r"35\%")
                model = RecordingChatModel(responses=[AIMessage(content=bad), AIMessage(content=LATEX), AIMessage(content=PASS)])
                saved = run_validated_pipeline("Required: Python SQL backend engineering.", SOURCE, root / "run", model=model, vector_store=store, require_pdf=True)
            report = json.loads(saved["validation"].read_text())
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["attempts"][-1]["compilation"], "PASS")
            self.assertEqual(report["attempts"][-1]["grounding"], "PASS")
            self.assertTrue(saved["pdf"].read_bytes().startswith(b"%PDF-"))
            if shutil.which("pdftotext"):
                text = subprocess.check_output(["pdftotext", str(saved["pdf"]), "-"], text=True)
                self.assertIn("Demo Candidate", text)
                self.assertIn("2027", text)
                self.assertIn("18%", text)
                self.assertIn("Python", text)
                self.assertNotIn("35%", text)
            self.assertIn(r"18\%", saved["latex"].read_text())
            self.assertNotIn("moderncv", saved["latex"].read_text())
            evidence = saved["evidence"].read_text()
            self.assertIn("experience.md", evidence)
            self.assertIn("career_item:1:chunk:0", evidence)
            self.assertTrue((saved["latex"].parent / "trace.json").exists())
            if not agent_mode:
                self.assertEqual(len(report["attempts"]), 2)
                self.assertIn("retrieved_candidate_evidence", str(model.seen[0]))

    def test_explicit_rag_pipeline_repairs_and_compiles(self):
        self.run_pipeline()

    def test_agent_graph_retrieves_writes_validates_and_compiles(self):
        self.run_pipeline(agent_mode=True)

    @unittest.skipUnless(os.getenv("RESUME_LIVE_EMBEDDINGS") == "1", "Opt-in MiniLM download required")
    def test_real_minilm_index_to_compiled_pdf(self):
        self.run_pipeline(live_embeddings=True)


if __name__ == "__main__":
    unittest.main()
