"""Configure this application's Hugging Face cache before importing HF libraries."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Keep model downloads in a writable project folder. This only changes the
# current Python process; it does not modify Windows or the user's global cache.
CACHE_HOME = Path(os.environ.get(
    "RESUME_MODEL_CACHE", str(PROJECT_ROOT / "backend" / "data" / "huggingface")
)).expanduser()
if not CACHE_HOME.is_absolute():
    CACHE_HOME = PROJECT_ROOT / CACHE_HOME
CACHE_HOME = CACHE_HOME.resolve()
MODEL_CACHE_DIRECTORY = CACHE_HOME / "hub"

# Hub/Xet cache locations are read at import time, so this module must be
# imported before langchain_huggingface, transformers, or huggingface_hub.
for name, directory in {
    "HF_HOME": CACHE_HOME,
    "HF_HUB_CACHE": MODEL_CACHE_DIRECTORY,
    "HF_XET_CACHE": CACHE_HOME / "xet",
    "HF_ASSETS_CACHE": CACHE_HOME / "assets",
    "SENTENCE_TRANSFORMERS_HOME": MODEL_CACHE_DIRECTORY,
}.items():
    os.environ[name] = str(directory)


def get_model_cache_directory() -> str:
    """Create the model cache when needed, with an actionable error on failure."""
    try:
        MODEL_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OSError(
            f"Cannot create the model cache at {MODEL_CACHE_DIRECTORY}. "
            "Set RESUME_MODEL_CACHE in .env to a folder you can write to, "
            "then restart the Python process."
        ) from error
    return str(MODEL_CACHE_DIRECTORY)
