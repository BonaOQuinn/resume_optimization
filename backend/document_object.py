from langchain_core.documents import Document
from backend.prepare_documents import prepare_documents

prepared = prepare_documents()

resume_documents = [
    Document(
        id=record["id"],
        page_content=record["text"],
        metadata=record["metadata"],
    )
    for record in prepared
]

