"""Build LangChain Document objects from the prepared resume records."""

import argparse
from pathlib import Path
from typing import Any, Mapping

from langchain_core.documents import Document

from backend.prepare_documents import DEFAULT_DATABASE, prepare_documents


def to_langchain_documents(records: list[Mapping[str, Any]]) -> list[Document]:
    """Convert prepared records to LangChain documents, keeping stable IDs.

    Null metadata fields are omitted because Chroma metadata values should be
    scalar values; the source values remain available in the SQLite database.
    """
    documents = []
    for record in records:
        metadata = record["metadata"]
        documents.append(
            Document(
                id=str(record["id"]),
                page_content=str(record["text"]),
                metadata={key: value for key, value in metadata.items() if value is not None},
            )
        )
    return documents


def create_resume_documents(database_path: Path = DEFAULT_DATABASE) -> list[Document]:
    """Load all resume records from SQLite and convert them to LangChain documents."""
    records = prepare_documents(database_path)
    return to_langchain_documents(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args()

    documents = create_resume_documents(args.database)
    print(f"Prepared {len(documents)} resume documents.")
    for document in documents:
        print(f"{document.id}: {document.metadata.get('item_type')} — "
              f"{document.metadata.get('title') or document.metadata.get('name')}")


if __name__ == "__main__":
    main()
