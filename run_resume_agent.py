"""Invoke the resume workflow with local source files and save its returned files."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from agent import create_resume_agent, get_resume_vector_store, prepare_resume_request, initialize_chat_model
from deepagents.backends.utils import file_data_to_string
from direct_resume import run_direct_resume_optimization
from resume_pipeline import run_validated_pipeline
from resume_validation import validate_and_export
from prompts import EVIDENCE_PATH, LATEX_PATH


def run_resume_optimization(
    job_document: str,
    base_resume: str,
    output_directory: Path,
    *,
    model=None,
    require_pdf=False,
) -> dict[str, Path]:
    """Run the agent with up to two validated corrections before export.

    The output directory must not already exist. Saved LaTeX is a draft: neither
    a filesystem write nor a successful invocation proves factual validity.
    """
    request = prepare_resume_request(job_document, base_resume)
    output_directory = Path(output_directory).resolve()
    if output_directory.exists():
        raise FileExistsError(f"Use a new output directory: {output_directory}")
    if model is None or isinstance(model, str):
        model = initialize_chat_model(model)
    compiled = create_resume_agent(model=model)

    def generate(issues):
        nonlocal request
        if issues:
            request["messages"] = list(request["messages"]) + [{"role": "user", "content": "Correct these validation failures and write the complete corrected LaTeX to /output/optimized_resume.tex using the supplied inputs:\n" + "\n".join(issues)}]
            request["files"] = {p: d for p, d in request.get("files", {}).items() if p != LATEX_PATH}
        result = compiled.invoke(request, config={"recursion_limit": 150})
        files = result.get("files", {})
        # Carry conversation/evidence, not graph-internal termination state.
        request = {"messages": list(result["messages"]), "files": files}
        return {
            "response": result["messages"][-1].text,
            "latex": file_data_to_string(files[LATEX_PATH]) if LATEX_PATH in files else None,
            "evidence": file_data_to_string(files[EVIDENCE_PATH]) if EVIDENCE_PATH in files else None,
            "sources": [file_data_to_string(data) for path, data in files.items() if path.startswith("/retrieved/resume/")],
            "trace": [message.model_dump(mode="json") for message in result["messages"]],
        }

    return validate_and_export(output_directory, generate, model, base_resume, require_pdf=require_pdf)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-file", type=Path, required=True)
    parser.add_argument("--resume-file", type=Path, required=True,
                        help="Complete source resume, including contact, education, and dates; .md or .tex.")
    parser.add_argument("--model", help="provider:model; defaults to RESUME_CHAT_MODEL in .env")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--direct", action="store_true",
                        help="Test one-call LaTeX generation using the full source resume; skips retrieval and subagents.")
    modes.add_argument("--pipeline", action="store_true",
                       help="Run the default job-tailored pipeline with retrieval, selection, review, and compilation.")
    modes.add_argument("--agent", action="store_true", help="Use the legacy autonomous agent workflow.")
    parser.add_argument("--require-pdf", action="store_true",
                        help="Require successful pdflatex compilation before exporting an accepted draft.")
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
        if args.direct:
            print("Running direct LaTeX generation test (no retrieval or subagents).", flush=True)
            saved = run_direct_resume_optimization(job, resume, output_directory, model=args.model, require_pdf=args.require_pdf)
        elif not args.agent:
            print("Running job-tailored RAG pipeline. Local inference may take several minutes.", flush=True)
            saved = run_validated_pipeline(job, resume, output_directory, model=args.model, require_pdf=args.require_pdf,
                                           progress=lambda stage, message: print(f"[{stage}] {message}", flush=True))
        else:
            if not get_resume_vector_store().get(limit=1, include=[])["ids"]:
                raise ValueError("The resume collection is empty. Run python main.py first.")
            print("Running resume agent. Local inference may take several minutes.", flush=True)
            saved = run_resume_optimization(job, resume, output_directory, model=args.model, require_pdf=args.require_pdf)
    except Exception as error:
        parser.exit(1, f"Resume run failed: {error}\nRun diagnostics: {output_directory}\n")
    print(f"Agent response saved to {saved['response']}")
    if "latex" not in saved:
        parser.exit(2, "Validation did not pass. Read validation.json and response.md; no accepted draft was exported.\n")
    print(f"LaTeX draft saved to {saved['latex']}")
    print(f"Validation report: {saved['validation']}")
    if "pdf" in saved:
        print(f"Compiled PDF saved to {saved['pdf']}")
    else:
        print("Compilation was not run: pdflatex was not found. Use --require-pdf to require it.")
    print("Review source facts and the validation report before use; LLM grounding review is not a factual guarantee.")


if __name__ == "__main__":
    main()

