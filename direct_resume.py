"""Single-writer generation with shared validation and bounded correction."""

import json
from pathlib import Path

from prompts import DEFAULT_GUIDELINES_PATH, DEFAULT_TEMPLATE_PATH, FACTUALITY_RULES
from resume_validation import extract_complete_latex, validate_and_export


def run_direct_resume_optimization(
    job_document, base_resume, output_directory, *, model=None,
    require_pdf=False, evidence_documents=None,
):
    for name, value in (("job_document", job_document), ("base_resume", base_resume)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must contain source text.")
    if Path(output_directory).exists():
        raise FileExistsError(f"Use a new output directory: {output_directory}")
    evidence_documents = evidence_documents or []
    instructions = (
        "Write the requested complete LaTeX resume from the candidate source facts. "
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
        + DEFAULT_TEMPLATE_PATH.read_text(encoding="utf-8")
    )
    inputs = {"job_posting": job_document, "source_resume": base_resume,
              "retrieved_candidate_evidence": evidence_documents}
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": "Create the resume from this reference data:\n" + json.dumps(inputs, ensure_ascii=False)},
    ]
    if model is None or isinstance(model, str):
        from agent import initialize_chat_model
        model = initialize_chat_model(model)

    last_response = None

    def generate(issues):
        nonlocal last_response
        attempt_messages = list(messages)
        if issues:
            attempt_messages.extend([
                {"role": "assistant", "content": last_response},
                {"role": "user", "content": "Correct these validation failures using the original source facts and approved template. Return the complete corrected LaTeX.\n" + "\n".join(issues)},
            ])
        response = model.invoke(attempt_messages).text
        last_response = response
        return {"response": response, "sources": evidence_documents,
                "evidence": "\n\n".join(evidence_documents),
                "trace": attempt_messages[1:] + [{"role": "assistant", "content": response}]}

    return validate_and_export(output_directory, generate, model, base_resume, require_pdf=require_pdf)
