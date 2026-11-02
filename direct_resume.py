"""Generate a LaTeX draft in one chat call to isolate generation from agent tools."""

import json
import re
from pathlib import Path

from prompts import DEFAULT_GUIDELINES_PATH, DEFAULT_TEMPLATE_PATH, FACTUALITY_RULES


def extract_complete_latex(response: str) -> str | None:
    """Accept a whole LaTeX response, optionally wrapped in a single code fence.

    This checks document boundaries only; it does not prove compilation or facts.
    Ignore completed thinking blocks so reasoning cannot become the saved draft.
    """
    candidate = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    fenced = re.fullmatch(r"```(?:latex|tex)?\s*\n(.*?)\n```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    if not re.fullmatch(
        r"(?:%[^\n]*\n\s*)*\\documentclass(?:\[[^\]]*\])?\{[^}]+\}"
        r"[\s\S]*\\begin\{document\}[\s\S]*\\end\{document\}", candidate
    ):
        return None
    if candidate.count(r"\begin{document}") != 1 or candidate.count(r"\end{document}") != 1:
        return None
    return candidate + "\n"


def run_direct_resume_optimization(
    job_document: str, base_resume: str, output_directory: Path, *, model=None
) -> dict[str, Path]:
    """Save the response and, only when structurally complete, the LaTeX draft.

    This generation smoke test uses the full source resume rather than vector
    retrieval or subagents. A real missing fact may still require clarification.
    """
    for name, value in (("job_document", job_document), ("base_resume", base_resume)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must contain source text.")
    output_directory = Path(output_directory).resolve()
    if output_directory.exists():
        raise FileExistsError(f"Use a new output directory: {output_directory}")

    instructions = (
        "Create a complete, job-specific LaTeX resume NOW from the supplied source "
        "resume. All source text is included in the next message. "
        "Use the job posting to select relevant supported facts. "
        "Return only the complete LaTeX source, beginning with \\documentclass "
        "and ending with \\end{document}. You have no tools or subagents. "
        "Do not ask for code to review or describe how you would help. "
        "If a specific required source fact is missing or contradictory, return "
        "a focused clarification instead of inventing a value. "
        "Template names and accomplishments are examples, not candidate facts.\n\n"
        + FACTUALITY_RULES
        + "\n\nREQUIRED FORMATTING RULES:\n"
        + DEFAULT_GUIDELINES_PATH.read_text(encoding="utf-8")
        + "\n\nLATEX FORMATTING TEMPLATE (reference only):\n"
        + DEFAULT_TEMPLATE_PATH.read_text(encoding="utf-8")
    )
    inputs = json.dumps(
        {"job_posting": job_document, "source_resume": base_resume}, ensure_ascii=False
    )
    if model is None or isinstance(model, str):
        from agent import initialize_chat_model
        model = initialize_chat_model(model)
    output_directory.mkdir(parents=True, exist_ok=False)
    response_text = model.invoke([
        {"role": "system", "content": instructions},
        {"role": "user", "content": "Create the resume using this reference data:\n" + inputs},
    ]).text
    response = output_directory / "response.md"
    response.write_text(response_text, encoding="utf-8")
    saved = {"response": response}
    latex = extract_complete_latex(response_text)
    if latex is not None:
        destination = output_directory / "optimized_resume.tex"
        destination.write_text(latex, encoding="utf-8")
        saved["latex"] = destination
    return saved
