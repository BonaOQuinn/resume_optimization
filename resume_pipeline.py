"""Requirement retrieval -> cited selection -> tailored writer -> review -> PDF."""

import json
from collections import defaultdict
from pathlib import Path

from direct_resume import run_direct_resume_optimization
from tailoring import job_requirements, build_tailoring_plan


def retrieve_evidence(job_document, vector_store, *, requirements=None, with_sources=False, progress=None):
    """Search every requirement with the existing MiniLM collection.

    Rank by reciprocal rank across searches, preserve category diversity, and
    retain evidence from late requirements instead of the first N posting lines.
    Retrieval signals relevance only; the cited plan must establish actual support.
    """
    requirements = requirements or job_requirements(job_document)
    documents, scores, matches = {}, defaultdict(float), {}
    first_matches = defaultdict(set)
    for index, requirement in enumerate(requirements):
        if progress:
            progress("retrieving", f"Finding evidence for requirement {index + 1}/{len(requirements)}")
        query = requirement["text"]
        found = []
        for category in ("experience", "project", "skill", "course"):
            for rank, doc in enumerate(vector_store.similarity_search(query, k=2, filter={"item_type": category})):
                key = str(doc.id or json.dumps([doc.metadata, doc.page_content], sort_keys=True))
                documents[key] = doc
                scores[key] += 1 / (rank + 1)
                if rank == 0:
                    first_matches[key].add((requirement["id"], category))
                found.append(key)
        matches[requirement["id"]] = found
    # Reserve balanced categories, then greedily cover top-ranked matches not
    # represented yet. A rare late requirement must not lose to generic evidence
    # simply because generic evidence has a higher aggregate score.
    selected = []
    for category in ("experience", "project", "skill", "course"):
        ranked = sorted((key for key, doc in documents.items() if doc.metadata.get("item_type") == category),
                        key=lambda key: (-scores[key], key))
        for key in ranked[:4]:
            if key not in selected:
                selected.append(key)
    covered = set().union(*(first_matches[key] for key in selected))
    remaining = set(documents) - set(selected)
    while remaining and len(selected) < 32:
        best = max(remaining, key=lambda key: (len(first_matches[key] - covered), scores[key], key))
        if not first_matches[best] - covered:
            break
        selected.append(best)
        covered.update(first_matches[best])
        remaining.remove(best)
    sources = {
        key: f"Chunk ID: {key}\nMetadata: {json.dumps(documents[key].metadata, ensure_ascii=False)}\n{documents[key].page_content}"
        for key in selected
    }
    if with_sources:
        return sources, matches
    return list(sources.values())


def run_validated_pipeline(job_document, base_resume, output_directory, *, model=None, vector_store=None, require_pdf=False, progress=None, compile_pdf=True):
    from agent import prepare_resume_request, get_resume_vector_store, initialize_chat_model, check_chat_model_available
    from resume_validation import find_latex_compiler, candidate_source_text
    prepare_resume_request(job_document, base_resume)
    if require_pdf and not compile_pdf:
        raise ValueError("Required PDF output cannot be combined with disabled compilation.")
    if len(job_document) > 30000 or len(base_resume) > 60000:
        raise ValueError("Use a job description under 30,000 characters and source resume under 60,000 characters.")
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "validation.json"
    report_path.write_text(json.dumps({"status": "RUNNING", "stage": "preflight", "attempts": []}), encoding="utf-8")

    def emit(stage, message):
        (output / "progress.json").write_text(json.dumps({"stage": stage, "message": message}), encoding="utf-8")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["stage"] = stage
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if progress:
            progress(stage, message)

    try:
        emit("preflight", "Checking the resume, model, and PDF compiler")
        if require_pdf and not find_latex_compiler():
            raise ValueError("PDF output requires pdflatex. Restart VS Code after adding MiKTeX to PATH, or select LaTeX only.")
        if model is None or isinstance(model, str):
            check_chat_model_available(model)
            model = initialize_chat_model(model)
        if vector_store is None:
            emit("index", "Loading your existing MiniLM resume index")
            vector_store = get_resume_vector_store()
        if not vector_store.get(limit=1, include=[])["ids"]:
            raise ValueError("The resume collection is empty. Run python main.py to index your experience first.")
        requirements = job_requirements(job_document)
        sources, search_map = retrieve_evidence(job_document, vector_store, requirements=requirements, with_sources=True, progress=emit)
        if not sources:
            raise ValueError("No evidence was retrieved. Rebuild the resume index with python main.py.")
        (output / "resume_evidence.md").write_text("\n\n".join(sources.values()), encoding="utf-8")
        (output / "retrieval.json").write_text(json.dumps({"requirements": requirements, "search_matches": search_map, "selected_sources": list(sources)}, indent=2, ensure_ascii=False), encoding="utf-8")
        plan = build_tailoring_plan(model, job_document, candidate_source_text(base_resume), sources, requirements, output, emit)
        plan_path = output / "tailoring_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        saved = run_direct_resume_optimization(
            job_document, base_resume, output, model=model, require_pdf=require_pdf,
            evidence_documents=list(sources.values()), tailoring_plan=plan,
            progress=emit, _directory_ready=True, compile_pdf=compile_pdf,
        )
        saved["plan"] = plan_path
        saved["retrieval"] = output / "retrieval.json"
        return saved
    except BaseException as error:
        # Keep diagnostics even when the user interrupts a long local model call.
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.update(status="INTERRUPTED" if not isinstance(error, Exception) else "ERROR", error=str(error) or "Run interrupted")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        emit("error", str(error) or "Run interrupted")
        raise
