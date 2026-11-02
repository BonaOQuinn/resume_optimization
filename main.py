"""Create resume embeddings and store them in the local Chroma database."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

# Configure caches before importing libraries that read Hugging Face settings.
from backend.model_cache import get_model_cache_directory

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from backend.chunk_documents import EMBEDDING_MODEL, split_resume_documents
from backend.document_object import create_resume_documents
from backend.prepare_documents import DEFAULT_DATABASE


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PERSIST_DIRECTORY = PROJECT_ROOT / "backend" / "data" / "chroma"
COLLECTION_NAME = "resume_minilm_l6_v2_chunks"


def create_vector_store(persist_directory: Path = DEFAULT_PERSIST_DIRECTORY) -> Chroma:
    """Open (or create) the persistent Chroma collection for resume documents."""
    load_dotenv(PROJECT_ROOT / ".env")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        cache_folder=get_model_cache_directory(),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(Path(persist_directory).resolve()),
    )


def index_resume_documents(
    database_path: Path = DEFAULT_DATABASE,
    persist_directory: Path = DEFAULT_PERSIST_DIRECTORY,
) -> int:
    """Split and embed resume documents, returning the number of stored chunks.

    Running again updates matching chunk IDs and removes surplus chunks for
    the indexed parents after the new embeddings are stored successfully.
    This does not remove parents that have been deleted from SQLite.
    """
    database_path = Path(database_path).resolve()
    if not database_path.is_file():
        raise FileNotFoundError(
            f"Resume database not found: {database_path}. "
            "Create and populate the SQLite database before indexing."
        )

    documents = create_resume_documents(database_path)
    if not documents:
        return 0

    chunks = split_resume_documents(documents)
    if not chunks:
        return 0
    vector_store = create_vector_store(persist_directory)
    # Clean up shortened documents only after the replacement chunks are saved.
    previous_ids = vector_store.get(
        where={"parent_id": {"$in": [str(document.id) for document in documents]}},
        include=[],
    )["ids"]
    chunk_ids = [str(chunk.id) for chunk in chunks]
    vector_store.add_documents(documents=chunks, ids=chunk_ids)
    obsolete_ids = sorted(set(previous_ids) - set(chunk_ids))
    if obsolete_ids:
        vector_store.delete(ids=obsolete_ids)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--persist-directory", type=Path, default=DEFAULT_PERSIST_DIRECTORY)
    args = parser.parse_args()

    try:
        count = index_resume_documents(args.database, args.persist_directory)
    except Exception as error:
        parser.exit(1, f"Resume indexing failed: {error}\n")

    if count == 0:
        print("No resume documents found; nothing was added to Chroma.")
    else:
        print(
            f"Embedded and stored {count} resume chunks in the "
            f"'{COLLECTION_NAME}' collection at {args.persist_directory.resolve()}."
        )


if __name__ == "__main__":
    main()
