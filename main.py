"""Create resume embeddings and store them in the local Chroma database."""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from backend.document_object import create_resume_documents
from backend.prepare_documents import DEFAULT_DATABASE


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PERSIST_DIRECTORY = PROJECT_ROOT / "backend" / "data" / "chroma"
COLLECTION_NAME = "resume_collection"
EMBEDDING_MODEL = "text-embedding-3-large"


def create_vector_store(persist_directory: Path = DEFAULT_PERSIST_DIRECTORY) -> Chroma:
    """Open (or create) the persistent Chroma collection for resume documents."""
    load_dotenv(PROJECT_ROOT / ".env")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


def index_resume_documents(
    database_path: Path = DEFAULT_DATABASE,
    persist_directory: Path = DEFAULT_PERSIST_DIRECTORY,
) -> int:
    """Embed prepared resume documents and add them to the persistent collection.

    The prepared document IDs are passed to Chroma so records keep their stable
    source IDs when the indexing command is run again.
    """
    documents = create_resume_documents(database_path)
    if not documents:
        return 0

    vector_store = create_vector_store(persist_directory)
    vector_store.add_documents(
        documents=documents,
        ids=[str(document.id) for document in documents],
    )
    return len(documents)


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
            f"Embedded and added {count} resume documents to the "
            f"'{COLLECTION_NAME}' collection at {args.persist_directory.resolve()}."
        )


if __name__ == "__main__":
    main()
