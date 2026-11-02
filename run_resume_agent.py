"""Invoke the resume workflow with local source files and save its returned files."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from agent import create_resume_agent, get_resume_vector_store, prepare_resume_request
from deepagents.backends.utils import file_data_to_string
from prompts import EVIDENCE_PATH, LATEX_PATH


def run_resume_optimization(
    job_document: str,
    base_resume: str,
    output_directory: Path,
    *,
    model=None,
) -> dict[str, Path]:
    """Run once and export only known output paths from the virtual filesystem.

    The output directory must not already exist. Saved LaTeX is a draft: neither
    a filesystem write nor a successful invocation proves factual or PDF validity.
    """
    request = prepare_resume_request(job_document, base_resume)
    compiled = create_resume_agent(model=model)
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    result = compiled.invoke(request, config={"recursion_limit": 150})

    saved = {}
    response = output_directory / "response.md"
    response.write_text(result["messages"][-1].text, encoding="utf-8")
    saved["response"] = response
    # Do not turn arbitrary agent filesystem paths into real host paths.
    for key, virtual_path, filename in (
        ("latex", LATEX_PATH, "optimized_resume.tex"),
        ("evidence", EVIDENCE_PATH, "resume_evidence.md"),
    ):
        data = result.get("files", {}).get(virtual_path)
        if data is not None:
            text = file_data_to_string(data)
            if text.strip():
                destination = output_directory / filename
                destination.write_text(text, encoding="utf-8")
                saved[key] = destination
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-file", type=Path, required=True)
    parser.add_argument("--resume-file", type=Path, required=True,
                        help="Complete source resume, including contact, education, and dates; .md or .tex.")
    parser.add_argument("--model", help="provider:model; defaults to RESUME_CHAT_MODEL in .env")
    parser.add_argument("--output-dir", type=Path,
                        help="New directory for this run; existing directories are rejected.")
    args = parser.parse_args()
    output_directory = args.output_dir or (
        Path(__file__).resolve().parent / "outputs"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    try:
        job = args.job_file.read_text(encoding="utf-8-sig")
        resume = args.resume_file.read_text(encoding="utf-8-sig")
        prepare_resume_request(job, resume)  # Fail on missing text before loading models.
        if output_directory.exists():
            raise FileExistsError(f"Use a new output directory: {output_directory}")
        if not get_resume_vector_store().get(limit=1, include=[])["ids"]:
            raise ValueError("The resume collection is empty. Run python main.py first.")
        print("Running resume agent. Local inference may take several minutes.", flush=True)
        saved = run_resume_optimization(job, resume, output_directory, model=args.model)
    except Exception as error:
        parser.exit(1, f"Resume run failed: {error}\n")
    print(saved["response"].read_text(encoding="utf-8"))
    print(f"Agent response saved to {saved['response']}")
    if "latex" not in saved:
        parser.exit(2, "No LaTeX draft was returned. Read the response for questions or blockers.\n")
    print(f"LaTeX draft saved to {saved['latex']}")
    print("Review the agent response and draft before use. LaTeX compilation was not performed.")


if __name__ == "__main__":
    main()
