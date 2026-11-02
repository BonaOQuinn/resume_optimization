"""Single-writer generation with shared validation and bounded correction."""

import json
from pathlib import Path

from prompts import DEFAULT_GUIDELINES_PATH, DEFAULT_TEMPLATE_PATH, FACTUALITY_RULES
from resume_validation import extract_complete_latex, validate_and_export, candidate_source_text, active_tex


def run_direct_resume_optimization(
    job_document, base_resume, output_directory, *, model=None,
    require_pdf=False, evidence_documents=None, tailoring_plan=None,
    progress=None, _directory_ready=False, compile_pdf=True,
):
    for name, value in (("job_document", job_document), ("base_resume", base_resume)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must contain source text.")
    if Path(output_directory).exists() and not _directory_ready:
        raise FileExistsError(f"Use a new output directory: {output_directory}")
    evidence_documents = evidence_documents or []
    instructions = (
        "Write a job-specific LaTeX resume from the candidate source facts. "
        "Use the selection plan to prioritize relevant bullets within each role, "
        "select applicable projects, and order supported skills for the target job. "
        "Rewrite verbose bullets to highlight the supported responsibility or result "
        "that matters for this employer. Do not simply reproduce the base resume. "
        "Follow the plan's project/course choices unless they contradict the source "
        "facts or formatting rules. Keep internal gaps, citations, and planning notes "
        "outside the resume. Never add a requested qualification to fill an evidence gap. "
        "You have no tools or subagents. All inputs are supplied below. "
        "Preserve the approved template\'s entire preamble exactly. Use its article "
        "class, centered header, resumeSubheading and resumeProjectHeading macros. "
        "Do not switch to moderncv. Do not add a logo. Use real LaTeX links and bold, "
        "never Markdown. Preserve dates, including future graduation dates; do not "
        "change them to Present without source support. Never invent metrics or "
        "interpret a malformed number as a percentage. Omit an ambiguous optional "
        "claim. The base resume controls identity, dates, and completion status; flag "
        "conflicts with retrieved evidence instead of silently choosing a version. "
        "Ask a specific question if a required fact is missing. "
        "Return complete LaTeX, not a review offer.\n\n" + FACTUALITY_RULES
        + "\n\nREQUIRED FORMATTING:\n" + DEFAULT_GUIDELINES_PATH.read_text(encoding="utf-8")
        + "\n\nAPPROVED TEMPLATE (layout only, not candidate facts):\n"
        + active_tex(DEFAULT_TEMPLATE_PATH.read_text(encoding="utf-8"))
    )
    writer_evidence = evidence_documents
    if tailoring_plan is not None:
        cited = {e["source_id"] for m in tailoring_plan["requirements"] for e in m["evidence"]}
        selected_names = tailoring_plan["projects"]["include"] + tailoring_plan["courses"]["include"]
        writer_evidence = [e for e in evidence_documents
                           if e.splitlines()[0].removeprefix("Chunk ID: ") in cited
                           or any(name.casefold() in e.casefold() for name in selected_names)]
    inputs = {"job_posting": job_document, "source_resume": candidate_source_text(base_resume),
              "retrieved_candidate_evidence": writer_evidence}
    if tailoring_plan is not None:
        inputs["selection_plan"] = tailoring_plan
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": "Create the resume from this reference data:\n" + json.dumps(inputs, ensure_ascii=False)},
    ]
    if model is None or isinstance(model, str):
        from agent import initialize_chat_model
        model = initialize_chat_model(model)

    def generate(issues):
        attempt_messages = list(messages)
        if issues:
            # Regenerate from the original sources and concrete feedback. A full
            # extra LaTeX draft can overflow a local model's context and cause it
            # to drop the source evidence or original job instructions.
            attempt_messages.extend([
                {"role": "user", "content": "Correct these validation failures using the original source facts and approved template. Return the complete corrected LaTeX.\n" + "\n".join(issues)},
            ])
        response = model.invoke(attempt_messages).text
        return {"response": response, "sources": evidence_documents,
                "evidence": "\n\n".join(evidence_documents),
                "trace": attempt_messages[1:] + [{"role": "assistant", "content": response}]}

    return validate_and_export(
        output_directory, generate, model, base_resume, require_pdf=require_pdf,
        job_document=job_document if tailoring_plan is not None else None,
        tailoring_plan=tailoring_plan, progress=progress,
        _directory_ready=_directory_ready, compile_pdf=compile_pdf,
    )
