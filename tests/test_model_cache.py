"""Regressions for stale or inaccessible default Hugging Face cache locations."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import model_cache


ROOT = Path(__file__).resolve().parents[1]


class ModelCacheTests(unittest.TestCase):
    def test_cache_settings_are_applied_before_hf_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            blocked = folder / "blocked-cache"
            blocked.write_text("This is a file, not a usable cache directory.")
            writable = folder / "writable-cache"
            for entrypoint in ("main", "backend.chunk_documents"):
                with self.subTest(entrypoint=entrypoint):
                    result = subprocess.run(
                        [sys.executable, "-c", """
import importlib
import os
from pathlib import Path
import sys

importlib.import_module(sys.argv[1])
from huggingface_hub import constants
from backend.model_cache import get_model_cache_directory

expected = Path(os.environ["RESUME_MODEL_CACHE"]).resolve()
assert Path(constants.HF_HOME) == expected
assert Path(constants.HF_HUB_CACHE) == expected / "hub"
assert Path(constants.HF_XET_CACHE) == expected / "xet"
assert Path(get_model_cache_directory()) == expected / "hub"
assert (expected / "hub").is_dir()
""", entrypoint],
                        cwd=ROOT,
                        env={
                            **os.environ,
                            "RESUME_MODEL_CACHE": str(writable),
                            "HF_HOME": str(blocked),
                            "HF_HUB_CACHE": str(blocked),
                            "HF_XET_CACHE": str(blocked),
                            "HF_HUB_OFFLINE": "1",
                        },
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(blocked.read_text(),
                             "This is a file, not a usable cache directory.")

    def test_unusable_project_cache_has_an_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache-file"
            path.write_text("not a folder")
            with patch.object(model_cache, "MODEL_CACHE_DIRECTORY", path):
                with self.assertRaisesRegex(OSError, "Set RESUME_MODEL_CACHE"):
                    model_cache.get_model_cache_directory()


if __name__ == "__main__":
    unittest.main()
