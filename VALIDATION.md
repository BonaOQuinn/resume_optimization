# Validation evidence

Validated on 2026-11-02 using Python 3.12 and the repository requirements,
CPU PyTorch, Deep Agents/LangGraph, Chroma, and an installed TeX Live compiler.

Result: all 61 tests passed with RESUME_LIVE_EMBEDDINGS=1, including the real
MiniLM integration test and actual PDF compilation/text checks. No tests were
skipped in that run. Writer and grounding-review responses were simulated.

## Checks actually exercised

| Layer | Validation |
| --- | --- |
| Source preparation | Actual SQLite schema, record reads, document conversion, and chunk IDs |
| Embeddings | Actual downloaded all-MiniLM-L6-v2 weights and tokenizer in the opt-in integration test |
| Persistence and retrieval | Actual persistent Chroma collection reopened from disk and similarity searched |
| Agent execution | Actual graph, search_resume tool, StateBackend writes, and export path |
| Validation | Template/preamble, sections/macros, Markdown, percentage escaping, supported numbers and contacts |
| Correction | Invalid template/metric and compilation failures corrected before export; bounded failures rejected |
| Review routing | Separate review call with strict verdict parsing; invalid decisions fail closed |
| Compilation | Actual pdflatex invocation; PDF bytes and extracted text checked |
| Failure handling | No accepted .tex/PDF on unresolved validation; provider errors recorded |

The full suite was run with:

```bash
RESUME_LIVE_EMBEDDINGS=1 python -m unittest discover -v
```

Chat-model writer and reviewer outputs in these tests are simulated. Tests prove
integration and validation behavior for the supplied synthetic responses; they
do not prove that a live chat model follows instructions or that LLM review
detects every unsupported claim. No private user resume was used in these tests.

## Live acceptance still required

Run on the computer that hosts Ollama, using the complete actual source resume
and populated resume database:

```powershell
python main.py
python run_resume_agent.py --pipeline --require-pdf --model ollama:qwen3:8b --job-file jobs/playstation.md --resume-file inputs/private/base_resume.tex
```

Acceptance requires exit 0, validation.json status PASS, final grounding PASS,
final compilation PASS, and both optimized_resume.tex and optimized_resume.pdf.
Inspect supported claims and layout before using the result. Exit 2 is a failed
acceptance run, even if response.md includes something that resembles a resume.

The hosted environment cannot access the user's local Ollama server or private
resume. Live inference with that configuration has not been verified here.

The GitHub Actions workflow runs the offline suite and real compilation on
pushes and pull requests; MiniLM downloading is an explicit local opt-in.
