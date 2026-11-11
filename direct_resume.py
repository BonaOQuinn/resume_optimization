"""Single-writer generation with shared validation and bounded correction."""

import json
from pathlib import Path

from prompts import DEFAULT_GUIDELINES_PATH, DEFAULT_TEMPLATE_PATH, FACTUALITY_RULES
from resume_validation import (extract_complete_latex, validate_and_export, candidate_source_text,
                               split_writer_notes, NOTES_MARKER)
from evidence_catalog import selected_name_supported
from tailoring import slim_plan


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
        "The application adds the approved Jake preamble (document class, packages "
        "and macro definitions) automatically. Return ONLY the document body: the "
        "content between \\begin{document} and \\end{document}, starting with the "
        "centered header. Do not write \\documentclass, \\usepackage, \\newcommand, "
        "or the begin/end document lines. Use the template's centered header, "
        "resumeSubheading and resumeProjectHeading macros. "
        "Do not switch to moderncv. Do not add a logo. Use real LaTeX links and bold, "
        "never Markdown. Escape percent signs as \\%. Preserve dates, including future graduation dates; do not "
        "change them to Present without source support. Never invent metrics or "
        "interpret a malformed number as a percentage. Omit an ambiguous optional "
        "claim. The base resume controls identity, dates, and completion status; flag "
        "conflicts with retrieved evidence instead of silently choosing a version. "
        "Return the complete resume body in LaTeX, not a review offer. The body must "
        "end with its last LaTeX line. If a source fact is ambiguous or looks like a "
        "typo, keep the base resume's wording in the resume and, after the body, add a "
        "line containing exactly " + NOTES_MARKER + " followed by short plain-text "
        "questions for the candidate. The application removes that part from the "
        "resume and shows it to the candidate. Add nothing else after the body.\n\n" + FACTUALITY_RULES
        + "\n\nREQUIRED FORMATTING:\n" + DEFAULT_GUIDELINES_PATH.read_text(encoding="utf-8")
        + "\n\nAPPROVED TEMPLATE BODY (layout example only, not candidate facts):\n"
        + candidate_source_text(DEFAULT_TEMPLATE_PATH.read_text(encoding="utf-8"))
    )
    writer_evidence = evidence_documents
    if tailoring_plan is not None:
        cited = {e["source_id"] for m in tailoring_plan["requirements"] for e in m["evidence"]}
        cited.update(r["source_id"] for section in ("projects", "courses")
                     for r in tailoring_plan[section].get("selected_records", []))
        writer_evidence = [e for e in evidence_documents
                           if e.splitlines()[0].removeprefix("Chunk ID: ") in cited
                           or any(selected_name_supported(name, section, {"chunk": e})
                                  for section in ("projects", "courses") for name in tailoring_plan[section]["include"])]
    inputs = {"job_posting": job_document, "source_resume": candidate_source_text(base_resume),
              "retrieved_candidate_evidence": writer_evidence}
    if tailoring_plan is not None:
        # The cited evidence text is already in retrieved_candidate_evidence;
        # repeating every quote in the plan roughly doubled the prompt.
        inputs["selection_plan"] = slim_plan(tailoring_plan)
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": "Create the resume from this reference data:\n" + json.dumps(inputs, ensure_ascii=False)},
    ]
    if model is None or isinstance(model, str):
        from agent import initialize_chat_model
        model = initialize_chat_model(model)

    # A full extra LaTeX draft can overflow a local model's context and cause it
    # to drop the source evidence or original job instructions.
    revise_in_place = type(model).__name__ != "ChatOllama"
    feedback = []  # Every issue raised in this run, oldest first.
    last_draft = []

    def generate(issues):
        attempt_messages = list(messages)
        if issues:
            earlier = [i for i in feedback if i not in issues]
            feedback.extend(i for i in issues if i not in feedback)
            correction = ("Correct these validation failures using the original source facts and approved template. "
                          "Return the complete corrected document body only.\n" + "\n".join(issues))
            if earlier:
                # A fresh rewrite used to reintroduce problems fixed one attempt earlier.
                correction += ("\n\nFeedback from earlier attempts still applies. Do not reintroduce these problems:\n"
                               + "\n".join(earlier))
            if revise_in_place and last_draft:
                # Editing the rejected draft converges; regenerating from scratch
                # gave the reviewers new wording to object to on every attempt.
                attempt_messages.append({"role": "assistant", "content": last_draft[-1]})
                correction = ("Revise your previous draft. Change only what the feedback requires and keep "
                              "every other line exactly as written.\n" + correction)
            attempt_messages.append({"role": "user", "content": correction})
        response = model.invoke(attempt_messages).text
        last_draft.append(split_writer_notes(response)[0])
        return {"response": response, "sources": evidence_documents,
                "evidence": "\n\n".join(evidence_documents),
                "trace": attempt_messages[1:] + [{"role": "assistant", "content": response}]}

    return validate_and_export(
        output_directory, generate, model, base_resume, require_pdf=require_pdf,
        job_document=job_document if tailoring_plan is not None else None,
        tailoring_plan=tailoring_plan, progress=progress,
        _directory_ready=_directory_ready, compile_pdf=compile_pdf,
    )
