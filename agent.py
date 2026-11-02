"""Resume evidence retrieval tools for a Deep Agent.

Pass both search_resume and this module's backend to create_deep_agent.
StateBackend file operations must run inside an agent/graph execution.
"""

import json
from functools import lru_cache
from typing import Literal

# Reuse the indexing configuration, including MiniLM, normalization, collection,
# and project-local Hugging Face caches. Import it before other HF integrations.
from main import create_vector_store

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data
from langchain.tools import tool
from langchain_core.utils.uuid import uuid7
from langchain_core.language_models import BaseChatModel

from prompts import BASE_RESUME_PATH, JOB_PATH, OutputMode, build_resume_prompts


backend = StateBackend()


@lru_cache(maxsize=1)
def get_resume_vector_store():
    """Open the existing resume collection once, without re-indexing documents."""
    return create_vector_store()


@tool(parse_docstring=True)
def search_resume(
    query: str,
    item_type: Literal["experience", "project", "course", "skill"] | None = None,
) -> str:
    """Search resume experience, projects, courses, and skills for relevant evidence.

    Save up to four matching chunks to the agent filesystem. Use focused
    queries describing job requirements or skills, and read the returned files
    before making claims about the candidate.

    Args:
        query: Natural language search for resume evidence relevant to a job requirement.
        item_type: Optional category filter: experience, project, course, or skill.

    Returns:
        Successfully saved file paths under /retrieved/resume/, or a status
        message when the query is empty, no chunks exist, or saving fails.
    """
    query = query.strip()
    if not query:
        return "Enter a job requirement or skill to search the resume."

    search_options = {"k": 4}
    if item_type is not None:
        search_options["filter"] = {"item_type": item_type}
    retrieved_docs = get_resume_vector_store().similarity_search(query, **search_options)
    if not retrieved_docs:
        if item_type is not None:
            return (
                f"No indexed resume chunks of type '{item_type}' found. "
                "Try another query or check that those source records were indexed."
            )
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


def create_resume_agent(
    model: str | BaseChatModel,
    *,
    latex_guidelines: str | None = None,
    output_mode: OutputMode = "latex_with_notes",
    latex_template: str | None = None,
):
    """Assemble the resume orchestrator and reviewers using a caller-chosen model."""
    if model is None:
        raise ValueError("Choose a chat model for the resume agent.")
    instructions, subagents = build_resume_prompts(
        latex_guidelines, output_mode=output_mode, latex_template=latex_template
    )
    return create_deep_agent(
        model=model,
        tools=[search_resume],
        backend=backend,
        system_prompt=instructions,
        subagents=subagents,
    )


def prepare_resume_request(job_document: str, base_resume: str) -> dict:
    """Attach source inputs to the virtual filesystem for an agent invocation."""
    for name, value in (("job_document", job_document), ("base_resume", base_resume)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must contain source text.")
    return {
        "messages": [{
            "role": "user",
            "content": (
                "Optimize my resume for the job in " + JOB_PATH
                + " using the source resume in " + BASE_RESUME_PATH
                + " and the supplied LaTeX guidelines. Read both input files first."
            ),
        }],
        "files": {
            JOB_PATH: create_file_data(job_document),
            BASE_RESUME_PATH: create_file_data(base_resume),
        },
    }
