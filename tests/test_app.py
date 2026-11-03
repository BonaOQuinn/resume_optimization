import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from tests.resume_fixture import SOURCE, LATEX
from tests.test_tailoring import plan_for

APP = Path(__file__).resolve().parents[1] / "app.py"


class AppTests(unittest.TestCase):
    def test_upload_takes_precedence_over_pasted_and_default_sources(self):
        from app import read_source
        class Upload:
            name, size = "main.tex", 20
            def getvalue(self):
                return b"Uploaded candidate"
        self.assertEqual(read_source(Upload(), "Pasted candidate"), "Uploaded candidate")
        self.assertEqual(read_source(None, "Pasted candidate"), "Pasted candidate")

    def test_empty_job_shows_actionable_error(self):
        app = AppTest.from_file(str(APP)).run()
        self.assertEqual(len(app.exception), 0)
        next(b for b in app.button if b.label == "Generate tailored resume").click().run()
        self.assertIn("Paste a job description", app.error[0].value)

    def test_job_and_pasted_source_reach_pipeline_and_downloads_persist(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            latex = output / "optimized_resume.tex"
            latex.write_text(LATEX)
            report = output / "validation.json"
            report.write_text(json.dumps({"status": "PASS", "attempts": []}))
            plan = plan_for()
            plan["requirements"][0]["text"] = "Python backend engineering"
            (output / "tailoring_plan.json").write_text(json.dumps(plan))
            def generate(job, source, destination, **kwargs):
                self.assertEqual(job, "Python backend engineering")
                self.assertEqual(source, SOURCE)
                self.assertFalse(kwargs["require_pdf"])
                # The app owns output paths; create actual reports in that run.
                destination.mkdir(parents=True)
                (destination / "validation.json").write_bytes(report.read_bytes())
                (destination / "tailoring_plan.json").write_text(json.dumps(plan))
                kwargs["progress"]("writing", "Writing the tailored resume")
                return {"latex": latex, "validation": destination / "validation.json"}
            with patch("resume_pipeline.run_validated_pipeline", side_effect=generate) as run:
                app = AppTest.from_file(str(APP)).run()
                app.text_area(key="source_text").set_value(SOURCE)
                app.text_area(key="job").set_value("Python backend engineering")
                app.checkbox(key="require_pdf").uncheck()
                next(b for b in app.button if b.label == "Generate tailored resume").click().run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any("LaTeX resume is ready" in s.value for s in app.success))
                self.assertEqual(run.call_count, 1)
                app.run()
                self.assertEqual(run.call_count, 1)  # Download/rerun does not regenerate.
                app.text_area(key="job").set_value("Different job").run()
                self.assertTrue(any("inputs have changed" in s.value for s in app.info))
                self.assertEqual(len(app.success), 0)

    def test_failed_run_never_shows_success(self):
        with patch("resume_pipeline.run_validated_pipeline", side_effect=ConnectionError("Ollama unavailable")):
            app = AppTest.from_file(str(APP)).run()
            app.text_area(key="source_text").set_value(SOURCE)
            app.text_area(key="job").set_value("Python backend engineering")
            next(b for b in app.button if b.label == "Generate tailored resume").click().run()
            self.assertEqual(len(app.success), 0)
            self.assertTrue(any("Ollama unavailable" in s.value for s in app.error))
