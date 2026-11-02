# Validation evidence

Validated on 2026-11-02 with Python 3.12, the repository dependencies,
Streamlit 1.66.1, CPU MiniLM, Chroma, and TeX Live.

Result: **75 tests passed, 1 skipped** (76 discovered). The skipped test is the
explicit live Ollama acceptance test. Actual MiniLM inference, Chroma retrieval,
and pdflatex compilation passed. Selection-plan, writer, and review responses
in the automated integration tests were simulated.

```bash
RESUME_LIVE_EMBEDDINGS=1 python -m unittest discover -v
```

## Checks exercised

| Layer | Evidence |
| --- | --- |
| Source preparation | Actual SQLite reads, documents, tokenization and chunk IDs |
| Embeddings | Actual downloaded `all-MiniLM-L6-v2` weights and tokenizer |
| Retrieval | Actual persisted Chroma; all requirement queries, short and late-posting requirements, type diversity |
| Selection plan | Complete requirement IDs, exact source quotations, known source IDs, explicit gaps, bounded invalid-plan repair |
| Tailoring | Reject unchanged substantive resumes; revise after failed relevance review before export |
| Factuality | Candidate facts separated from the job; commented template examples removed; template/section/metric/contact checks and grounding review |
| LaTeX | Reject doubled row breaks; actual compiler success/failure, saved logs, PDF bytes and extracted text |
| Interruption | RUNNING checkpoints and diagnostic .tex exist before long reviews; handled interruption records INTERRUPTED |
| UI | Streamlit AppTest: missing input, pasted source/job routing, error state, retained results, and hiding stale results after input changes |
| Existing agent mode | Actual graph execution, search tool, StateBackend writes, correction and export |

No private resume data was committed. No new embedding model or collection was
introduced. The tests use synthetic candidate facts.

## Live chat acceptance

The workspace had no Ollama server at `127.0.0.1:11434`, and it cannot call the
server on the user's Windows computer. Live Qwen generation quality and the
Windows MiKTeX environment were **not verified** here. Simulated model responses
do not establish that a live model will always produce an acceptable resume.

The added opt-in test exercises two different jobs against a single synthetic
candidate using real MiniLM, Chroma, the selected live chat model, and pdflatex:

```powershell
$env:RESUME_LIVE_CHAT_MODEL = "ollama:qwen3:8b"
python -m unittest tests.test_live_acceptance -v
```

It requires the backend job to select the Python API project and omit the
unrelated vision project; the vision job must do the reverse. Both must preserve
the source graduation date, pass grounding/tailoring review, and compile to PDF.

For the user's actual inputs:

```powershell
python main.py
python -m streamlit run app.py
```

Or use the CLI in RUN_AGENT.md. Acceptance requires `validation.json` status PASS,
final grounding PASS, tailoring PASS, and compilation PASS when PDF is requested.
The accepted files are `optimized_resume.tex` and `optimized_resume.pdf`.
Diagnostic `attempt_*.tex` files are not accepted exports. Inspect the actual
claims and layout; model review and numeric-presence checks are not guarantees.

Browser screenshot inspection was attempted, but the browser binary downloads
were not valid archives in this environment. UI behavior was verified with
Streamlit AppTest; visual desktop/mobile screenshot review remains unverified.
