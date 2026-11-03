# Validation evidence

Validated on 2026-11-03 with Python 3.12, the repository dependencies,
Streamlit 1.66.1, CPU MiniLM, Chroma, and TeX Live.

Result: **85 tests passed, 1 skipped** (86 discovered). The skipped test is the
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
| Selection plan | Catalog ID resolution to source-owned quotes/labels; complete requirement coverage; exact source IDs; bounded repair; screenshot-pattern course regression |
| Tailoring | Reject unchanged substantive resumes; revise after failed relevance review before export |
| Factuality | Candidate facts separated from the job; commented template examples removed; template/section/metric/contact checks and grounding review |
| LaTeX | Reject doubled row breaks; actual compiler success/failure, saved logs, PDF bytes and extracted text |
| Interruption | RUNNING checkpoints and diagnostic .tex exist before long reviews; handled interruption records INTERRUPTED |
| UI | Streamlit AppTest: missing input, pasted source/job routing, uploaded-source precedence, error state, retained results, and hiding stale results after input changes |
| Existing agent mode | Actual graph execution, search tool, StateBackend writes, correction and export |

No private resume data was committed. No new embedding model or collection was
introduced. The tests use synthetic candidate facts.

## Evidence-planning regression (2026-11-03)

The reported planner failure was reproducible: code/name strings such as
`CS577: intro to Data Science and Principles` did not occur contiguously in
retrieved records containing separate `Course code:` and `Course:` fields.
Human-readable quotations also differed from source LaTeX display syntax.

The planner now selects source-owned evidence/project/course IDs. The application
attaches original raw excerpts and canonical paired course labels. Legacy literal
quotes accept only display-format differences; invented IDs, changed percentages,
and mismatched course code/name pairs still fail. Planning attempts now save
specific validation errors plus the source catalog for diagnosis.

Eight focused regression tests cover those cases. The full integration tests now
use the ID-based plan format and actual compilation. An additional local check
validated all 32 excerpts, 4 projects, and 7 paired courses extracted from the
user's previously uploaded LaTeX/evidence artifacts. Those private artifacts were
not committed. The new screenshot alone did not include its raw planning response,
so the regression reproduces the observed data shape, not that exact model output.

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
