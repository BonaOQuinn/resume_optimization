# Run and validate the resume pipeline

Run from the project root with your virtual environment active. Install
dependencies with `python -m pip install -r requirements.txt` and keep your
local Ollama server available. Use a downloaded tool-capable chat model such as
the configured `ollama:qwen3:8b`; MiniLM is the embedding model.

Place your complete source resume at `inputs/private/base_resume.tex`.
This ignored file must contain your actual identity/contact, education,
employment dates, and source facts. The tracked Markdown sections alone do not
contain all of them. Your original inputs are never overwritten.

## Validated RAG run

Populate your existing resume SQLite database first, then index it:

```powershell
python main.py
python run_resume_agent.py --pipeline --require-pdf --model ollama:qwen3:8b --job-file jobs/playstation.md --resume-file inputs/private/base_resume.tex
```

`--pipeline` explicitly retrieves focused job evidence from the existing Chroma
index, including project and course searches. It deduplicates and bounds evidence
to at most 24 chunks, then runs a writer, deterministic validators, an independent
grounding-review call, and compilation. This keeps retrieval and review steps
under application control. A failed candidate gets at most two corrections.

`--require-pdf` requires `pdflatex` on PATH and successful compilation. Install a
LaTeX distribution containing the packages used in `templates/jakes_resume.tex`
(for example MiKTeX or TeX Live) on the computer running the command. The runner
disables shell escape. Without this flag it still compiles when `pdflatex` is
available, otherwise reports compilation as `NOT_RUN`.

Use your exact installed model tag, or set `RESUME_CHAT_MODEL` in `.env` and omit
`--model`. Ollama is initialized with temperature 0 and a 16384-token context.
Set `RESUME_OLLAMA_NUM_CTX` in `.env` to adjust context size for your available
memory. Each correction retains the original inputs and latest rejected draft,
rather than accumulating all previous drafts.

## Other generation modes

Omit `--pipeline` to use the original Deep Agents orchestrator and specialist
subagents. Its exported draft now passes the same external validation, grounding
review, correction limit, and compilation gate. Rejected virtual drafts are
removed before correction so stale files cannot count as a successful rewrite.

Use `--direct` to isolate writing from retrieval and subagents:

```powershell
python run_resume_agent.py --direct --require-pdf --model ollama:qwen3:8b --job-file jobs/playstation.md --resume-file inputs/private/base_resume.tex
```

Direct mode uses the full source resume, posting, template, and rules. It still
performs validation and a separate grounding review; it is not a full RAG test.
`--direct` and `--pipeline` are mutually exclusive.

## Acceptance and diagnostics

Every run creates a fresh ignored `outputs/<timestamp>/` folder; existing output
directories are rejected. Optional `--output-dir` must name a new directory.

- `response.md`: latest writer/agent response.
- `attempt_1.md` through `attempt_3.md`: attempted responses.
- `trace.json`: model/agent conversation for debugging.
- `resume_evidence.md`: retrieved evidence when available.
- `validation.json`: checks, reviewer decision, compiler status, and failures.
- `optimized_resume.tex`: exported only after validation and review pass.
- `optimized_resume.pdf`: exported only after actual successful compilation.

Exit 0 means the export gate passed. Exit 2 means validation or clarification
remains unresolved; inspect `validation.json` and `response.md`. Exit 1 means
an input/index/provider/execution error; an error during generation is recorded
in `validation.json`. No failed candidate gets an accepted .tex or PDF filename.

The deterministic gate preserves the active Jake article preamble and checks
section order/macros, Markdown contamination, percent escaping, supported
percentages/GPA/years/contact links and emails, and incomplete documents. It
extracts a single fenced LaTeX document when explanatory notes surround it.

The grounding reviewer separately checks claim attribution, dates, education,
contacts, tools, metrics, and completion status. A malformed or contradictory
review verdict fails closed. The job posting is never candidate evidence.
Future graduation dates are allowed when source-supported. Malformed source
metrics such as `35maintain` must not become `35%`.

These checks do not prove every claim is true. Numeric presence does not prove
correct attribution, LLM reviews can be wrong, and compilation does not prove
layout quality or ATS performance. Inspect the final resume and its sources.

## Reproducible tests

```powershell
python -m unittest discover -v
$env:RESUME_LIVE_EMBEDDINGS = "1"
python -m unittest tests.test_full_pipeline -v
```

The default suite is offline for inference, using synthetic candidate facts,
an in-memory tokenizer and test embedding provider where appropriate. Integration
tests use actual SQLite, persistent Chroma, similarity retrieval, the real agent
graph/backend, correction/validation/export code, and pdflatex. They also extract
PDF text when pdftotext is installed. Compiler-dependent tests are explicitly
skipped when pdflatex is missing.

The opt-in test downloads and uses actual MiniLM weights and tokenizer through
the same indexing path. Chat generation and review remain simulated in both
suites, so these are not claims about a live Ollama model's compliance. To verify
your live model, run the validated RAG command above with your actual inputs and
require a PASS report, grounding PASS, compilation PASS, and the exported PDF.

ChatGPT's hosted workspace cannot access the Ollama server running on your
computer or your private local resume. A real local acceptance run is still
required for that configuration. Outputs and traces contain personal data;
keep the ignored folders out of Git.

