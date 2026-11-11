"""Local resume workbench. Start with: python -m streamlit run app.py"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from resume_pipeline import run_validated_pipeline
from resume_validation import find_latex_compiler

ROOT = Path(__file__).resolve().parent
DEFAULT_RESUME = ROOT / "inputs" / "private" / "base_resume.tex"
DEFAULT_MODEL = "anthropic:claude-opus-5-5"
load_dotenv(ROOT / ".env")


def read_source(upload, pasted):
    if upload is not None:
        if upload.size > 100_000:
            raise ValueError("Upload a source resume smaller than 100 KB.")
        return upload.getvalue().decode("utf-8-sig")
    if pasted.strip():
        return pasted
    if DEFAULT_RESUME.is_file():
        return DEFAULT_RESUME.read_text(encoding="utf-8-sig")
    return ""


def fingerprint(job, source, model, require_pdf):
    return hashlib.sha256(json.dumps([job, source, model, require_pdf]).encode()).hexdigest()


def show_result(saved, output):
    report_path = output / "validation.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    if "latex" in saved and report.get("status") == "PASS":
        st.success("Your tailored resume is ready." if "pdf" in saved else "Your tailored LaTeX resume is ready. PDF compilation was not run.")
        col1, col2 = st.columns(2)
        col1.download_button("Download LaTeX", saved["latex"].read_bytes(), "optimized_resume.tex", "application/x-tex", width="stretch")
        if "pdf" in saved:
            col2.download_button("Download PDF", saved["pdf"].read_bytes(), "optimized_resume.pdf", "application/pdf", width="stretch")
        with st.expander("View LaTeX source"):
            st.code(saved["latex"].read_text(encoding="utf-8"), language="latex")
    else:
        st.error("This run needs attention. An accepted resume was not exported.")
        if report.get("error"):
            st.write(report["error"])
        attempts = report.get("attempts", [])
        if attempts:
            for issue in attempts[-1].get("issues", []):
                if issue != report.get("error"):
                    st.write("• " + issue)
        if "unverified_latex" in saved:
            st.warning("The draft passed every review but could not be compiled on this computer. "
                       "You can download the LaTeX and compile it elsewhere (for example Overleaf).")
            st.download_button("Download reviewed LaTeX (not compiled)", saved["unverified_latex"].read_bytes(),
                               "optimized_resume.tex", "application/x-tex")

    notes_path = output / "writer_notes.md"
    if notes_path.exists():
        # Ambiguous source facts the writer kept as written instead of guessing.
        with st.expander("Questions to confirm in your source resume", expanded=True):
            st.text(notes_path.read_text(encoding="utf-8"))

    plan_path = output / "tailoring_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        with st.expander("How your experience matches this job", expanded=True):
            rows = [{"Job requirement": m["text"], "Support": m["support"].capitalize(), "Resume decision": m["action"]}
                    for m in plan["requirements"] if m["support"] != "context"]
            st.dataframe(rows, hide_index=True, width="stretch")
            st.write("**Projects:** " + plan["projects"]["reason"])
            st.write("**Coursework:** " + plan["courses"]["reason"])
            st.caption("Matches are model-reviewed suggestions. Missing qualifications stay outside the resume.")
            st.download_button("Download evidence plan", plan_path.read_bytes(), "tailoring_plan.json", "application/json")
    with st.expander("Run details"):
        st.caption(str(output))
        st.json(report)
        if report_path.exists():
            st.download_button("Download validation report", report_path.read_bytes(), "validation.json", "application/json")
        catalog_path = output / "evidence_catalog.json"
        if catalog_path.exists():
            st.download_button("Download source evidence catalog", catalog_path.read_bytes(), catalog_path.name, "application/json")
        plans = sorted(output.glob("plan_attempt_*.md"))
        if plans:
            st.download_button("Download latest planning response", plans[-1].read_bytes(), plans[-1].name, "text/markdown")
        logs = sorted(output.glob("compile_attempt_*.log"))
        if logs:
            st.download_button("Download compiler log", logs[-1].read_bytes(), logs[-1].name, "text/plain")


def main():
    st.set_page_config(page_title="Resume Studio", page_icon="📄", layout="centered")
    st.title("Resume Studio")
    st.write("Turn a job description into a resume grounded in your experience.")
    st.caption("Your approved Jake’s Resume layout · Your recorded facts · A clear match to the role")

    with st.sidebar:
        st.subheader("Resume Studio")
        st.caption("Fill in the job description and press Generate. Inputs are only read when you "
                   "press Generate, so editing them cannot interrupt a run in progress.")
        if not find_latex_compiler():
            st.info("pdflatex is unavailable. Install MiKTeX or uncheck PDF to generate LaTeX only.")
        with st.expander("Update your experience index"):
            st.caption("After updating your SQLite resume database, refresh the existing MiniLM index.")
            if st.button("Refresh index"):
                try:
                    from main import index_resume_documents
                    with st.spinner("Updating your resume evidence…"):
                        count = index_resume_documents()
                    st.success(f"Indexed {count} evidence chunks.")
                except Exception as error:
                    st.error(str(error))

    # A form batches widget changes until submission. Without it, any keystroke
    # or click during a multi-minute generation reruns the script and aborts it.
    with st.form("generate_form", border=False):
        job = st.text_area("Job description", height=320, max_chars=30000,
                           placeholder="Paste the responsibilities, qualifications, and preferred skills from the posting…", key="job")
        with st.expander("Your source resume"):
            st.caption("Keep your contact details, education, dates, and current experience in this source. "
                       "Priority: uploaded file, then pasted text, then inputs/private/base_resume.tex.")
            upload = st.file_uploader("Upload a source resume", type=["tex", "md", "txt"])
            pasted = st.text_area("Or paste source text", height=180, key="source_text")
        with st.expander("Generation settings"):
            model = st.text_input("Chat model", value=os.getenv("RESUME_CHAT_MODEL") or DEFAULT_MODEL, key="model",
                                  help="provider:model, e.g. anthropic:claude-opus-5-5 or ollama:qwen3:8b")
            require_pdf = st.checkbox("Generate PDF too", value=True, key="require_pdf")
            st.caption("The existing all-MiniLM-L6-v2 embedding index is used for retrieval.")
        submitted = st.form_submit_button("Generate tailored resume", type="primary", width="stretch")
    source_error = None
    try:
        source = read_source(upload, pasted)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        source, source_error = "", str(error)
    if source_error:
        st.error(source_error)
    if upload is not None:
        st.caption(f"Source resume: uploaded file {upload.name}")
    elif pasted.strip():
        st.caption("Source resume: the text you pasted")
    elif DEFAULT_RESUME.is_file():
        st.caption("Source resume: inputs/private/base_resume.tex")
    current = fingerprint(job, source, model, require_pdf)
    if submitted:
        st.session_state.pop("result", None)
        if not job.strip():
            st.error("Paste a job description first.")
        elif not source.strip():
            st.error("Add your source resume in the sidebar, or save it at inputs/private/base_resume.tex.")
        elif not model.strip():
            st.error("Set your chat model in Generation settings.")
        else:
            output = ROOT / "outputs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            with st.status("Preparing your resume…", expanded=True) as status:
                stage_text = st.empty()
                def progress(stage, message):
                    stage_text.write(message)
                    status.update(label=message)
                try:
                    saved = run_validated_pipeline(job, source, output, model=model.strip(), require_pdf=require_pdf, compile_pdf=require_pdf, progress=progress)
                    success = "latex" in saved
                    status.update(label="Resume ready" if success else "Review needs attention", state="complete" if success else "error", expanded=False)
                except Exception as error:
                    saved = {}
                    status.update(label="Generation stopped", state="error", expanded=False)
                    st.error(str(error))
                st.session_state["result"] = {"saved": saved, "output": output, "fingerprint": current}
    result = st.session_state.get("result")
    if result:
        if result["fingerprint"] != current:
            st.info("Your inputs have changed. Generate again to get a resume for this description.")
        else:
            show_result(result["saved"], result["output"])


if __name__ == "__main__":
    main()
