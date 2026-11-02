"""Resume evidence retrieval tools for a Deep Agent.

Pass both search_resume and this module's backend to create_deep_agent.
StateBackend file operations must run inside an agent/graph execution.
"""

import json
from functools import lru_cache

# Reuse the indexing configuration, including MiniLM, normalization, collection,
# and project-local Hugging Face caches. Import it before other HF integrations.
from main import create_vector_store

from deepagents.backends import StateBackend
from langchain.tools import tool
from langchain_core.utils.uuid import uuid7


backend = StateBackend()


@lru_cache(maxsize=1)
def get_resume_vector_store():
    """Open the existing resume collection once, without re-indexing documents."""
    return create_vector_store()


@tool(parse_docstring=True)
def search_resume(query: str) -> str:
    """Search resume experience, projects, courses, and skills for relevant evidence.

    Save up to four matching chunks to the agent filesystem. Use focused
    queries describing job requirements or skills, and read the returned files
    before making claims about the candidate.

    Args:
        query: Natural language search for resume evidence relevant to a job requirement.

    Returns:
        Successfully saved file paths under /retrieved/resume/, or a status
        message when the query is empty, no chunks exist, or saving fails.
    """
    query = query.strip()
    if not query:
        return "Enter a job requirement or skill to search the resume."

    retrieved_docs = get_resume_vector_store().similarity_search(query, k=4)
    if not retrieved_docs:
        return "No indexed resume chunks found. Run python main.py to index the resume first."

    # Use the full UUID: its first eight characters alone are time-based and
    # can be identical across consecutive searches, causing file overwrites.
    batch_id = uuid7().hex
    uploads: list[tuple[str, bytes]] = []
    for index, doc in enumerate(retrieved_docs, start=1):
        path = f"/retrieved/resume/{batch_id}/chunk_{index}.md"
        metadata = doc.metadata
        title = metadata.get("title") or metadata.get("name") or "Resume chunk"
        content = (
            f"# Resume evidence: {title}\n"
            f"Source file: {metadata.get('source_file', 'unknown')}\n"
            f"Chunk ID: {doc.id or 'unknown'}\n\n"
            f"Metadata: {json.dumps(metadata, ensure_ascii=False)}\n\n"
            f"{doc.page_content}"
        )
        uploads.append((path, content.encode("utf-8")))

    responses = backend.upload_files(uploads)
    saved_paths = [response.path for response in responses if response.error is None]
    failed = [response for response in responses if response.error is not None]

    message = f"Saved {len(saved_paths)} resume evidence chunks:"
    if saved_paths:
        message += "\n" + "\n".join(saved_paths)
    if failed:
        message += "\nFailed to save:\n" + "\n".join(
            f"{response.path}: {response.error}" for response in failed
        )
    return message
