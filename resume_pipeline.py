"""Explicit retrieval -> writer -> validation -> grounding review -> PDF pipeline."""

import json
import re
from pathlib import Path

from direct_resume import run_direct_resume_optimization


def retrieve_evidence(job_document, vector_store):
    """Use focused posting lines, plus explicit project/course relevance searches."""
    lines = [re.sub(r"^\s*[-*#\d.)]+\s*", "", line).strip()
             for line in job_document.splitlines() if len(line.strip()) > 25]
    queries = list(dict.fromkeys(line[:500] for line in lines))[:8]
    if not queries:
        queries = [job_document.strip()[:500]]
    documents = {}
    for query, category in [(q, None) for q in queries] + [(queries[0], "project"), (queries[0], "course")]:
        options = {"k": 4}
        if category:
            options["filter"] = {"item_type": category}
        for doc in vector_store.similarity_search(query, **options):
            key = doc.id or json.dumps([doc.metadata, doc.page_content], sort_keys=True)
            # Reserve space for up to four projects and four courses, while
            # bounding the input sent to a local model to at most 24 chunks.
            if category or key in documents or len(documents) < 16:
                documents[key] = doc
    return [
        f"Chunk ID: {doc.id}\nMetadata: {json.dumps(doc.metadata, ensure_ascii=False)}\n{doc.page_content}"
        for doc in documents.values()
    ]


def run_validated_pipeline(job_document, base_resume, output_directory, *, model=None, vector_store=None, require_pdf=False):
    # Check inputs before touching models or the vector index.
    from agent import prepare_resume_request, get_resume_vector_store
    prepare_resume_request(job_document, base_resume)
    if Path(output_directory).exists():
        raise FileExistsError(f"Use a new output directory: {output_directory}")
    if vector_store is None:
        vector_store = get_resume_vector_store()
    if not vector_store.get(limit=1, include=[])["ids"]:
        raise ValueError("The resume collection is empty. Run python main.py first.")
    evidence = retrieve_evidence(job_document, vector_store)
    if not evidence:
        raise ValueError("No evidence was retrieved from the populated resume index.")
    return run_direct_resume_optimization(
        job_document, base_resume, output_directory, model=model,
        require_pdf=require_pdf, evidence_documents=evidence,
    )
