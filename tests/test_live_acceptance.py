"""Opt in with RESUME_LIVE_CHAT_MODEL=ollama:qwen3:8b.

Uses real MiniLM, Chroma, the configured local chat model, reviews, and pdflatex.
This intentionally runs two different jobs against one synthetic candidate.
"""
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from main import create_vector_store, index_resume_documents
from resume_pipeline import run_validated_pipeline
from tests.resume_fixture import SOURCE


@unittest.skipUnless(os.getenv("RESUME_LIVE_CHAT_MODEL"), "Opt-in local chat model required")
class LiveAcceptanceTests(unittest.TestCase):
    def test_different_jobs_select_different_supported_projects(self):
        # Synthetic facts avoid publishing or requiring a private resume in CI.
        source = SOURCE + """
Project: Basketball Tracker, 2026. Trained a YOLO model to track a basketball in video.
Used MediaPipe for pose estimation and prepared labeled video frames for evaluation.
Technical Skills: YOLO, MediaPipe, OpenCV.
"""
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            database = work / "resume.db"
            with sqlite3.connect(database) as con:
                con.executescript((root / "backend/schema.sql").read_text())
                con.execute("INSERT INTO career_items VALUES (1, 'project', 'Python API', NULL, NULL, 'synthetic.md', 0)")
                con.execute("INSERT INTO career_items VALUES (2, 'project', 'Basketball Tracker', NULL, NULL, 'synthetic.md', 1)")
                con.execute("INSERT INTO career_items VALUES (3, 'experience', 'Engineer', 'Example Company', NULL, 'synthetic.md', 2)")
                con.executemany("INSERT INTO bullets VALUES (?, ?, ?, ?)", [
                    (1, 1, "Built an API in Python.", 0),
                    (2, 2, "Trained a YOLO model to track a basketball in video. Used MediaPipe for pose estimation and prepared labeled video frames for evaluation.", 0),
                    (3, 3, "Built Python and SQL services; increased bookings by 18%.", 0),
                ])
            index_resume_documents(database, work / "chroma")
            store = create_vector_store(work / "chroma")
            outputs = []
            for name, job, selected, omitted in [
                ("backend", "Backend internship: Develop Python APIs and SQL services. Work on databases and backend application development.", "Python API", "Basketball Tracker"),
                ("vision", "Computer vision internship: Track objects in video with YOLO. Use MediaPipe for pose estimation. Prepare labeled video frames for evaluation.", "Basketball Tracker", "Python API"),
            ]:
                saved = run_validated_pipeline(job, source, work / name, model=os.environ["RESUME_LIVE_CHAT_MODEL"], vector_store=store, require_pdf=True,
                                               progress=lambda stage, msg: print(f"[{name}:{stage}] {msg}", flush=True))
                report = json.loads(saved["validation"].read_text())
                self.assertEqual(report["status"], "PASS", json.dumps(report, indent=2))
                self.assertEqual(report["attempts"][-1]["tailoring"], "PASS")
                self.assertEqual(report["attempts"][-1]["compilation"], "PASS")
                draft = saved["latex"].read_text()
                self.assertIn(selected, draft)
                self.assertNotIn(omitted, draft)
                self.assertIn("2027", draft)
                self.assertNotIn("Golang", draft)
                self.assertTrue(saved["pdf"].read_bytes().startswith(b"%PDF-"))
                outputs.append(draft)
            self.assertNotEqual(*outputs)
