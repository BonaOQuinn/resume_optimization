# Run and validate the resume pipeline

Install dependencies in the existing virtual environment:

```powershell
python -m pip install -r requirements.txt
```

Use `python -m streamlit run app.py` for the local UI, or run the CLI below.
The source resume supplies identity/contact details, education and dates absent
from the indexed Markdown. Keep it at `inputs/private/base_resume.tex`; this is
an ignored file. The populated SQLite database and MiniLM index remain unchanged.

## Job-tailored pipeline (default)

```powershell
python main.py
python run_resume_agent.py --require-pdf --model ollama:qwen3:8b --job-file jobs/playstation.md --resume-file inputs/private/base_resume.tex
```

The existing `--pipeline` flag remains accepted, but this is now the default.
The steps are:

1. Check input text and, when required, compiler availability.
2. Segment the whole posting into requirement units; search every unit by
   experience, project, skill, and course in the existing MiniLM Chroma index.
3. Build a bounded selection plan with exact candidate-source quotes, relevant
   projects/courses, and direct/partial/gap/context decisions. Invalid plans get
   at most two repairs. Retrieval similarity alone is not a supported claim.
4. Write the job-specific resume with that plan, the source facts, and Jake’s
   template. Preserve dates and completion status; rephrase/prioritize supported
   bullets and skills rather than copying the original resume wholesale.
5. Check the template and selected factual invariants, then separately review
   factual grounding and job relevance. A generic or unsupported draft gets at
   most two corrections before export is rejected.
6. Compile with pdflatex when available; `--require-pdf` makes it mandatory.

Keep the local Ollama server running. Use your exact installed tag or set
`RESUME_CHAT_MODEL` in `.env`. The existing context defaults to 16384 tokens;
`RESUME_OLLAMA_NUM_CTX` can increase it if your hardware permits. Progress messages
identify the active stage; local model calls may take several minutes.

## PDF setup

Install MiKTeX or TeX Live and the packages in `templates/jakes_resume.tex`.
`pdflatex` is found on PATH, or in standard Windows MiKTeX/MiKTeX2 folders.
`RESUME_PDFLATEX` can name the executable explicitly. Reopen VS Code after changing
Windows PATH. Missing required compilers fail before model inference.

The compiler runs without shell escape, with a 60-second timeout. MiKTeX's
interactive package installer is disabled: install missing packages through
MiKTeX Console according to the saved log. Without `--require-pdf`, the CLI still
compiles when possible and reports `NOT_RUN` if no compiler is available. The UI
can explicitly disable compilation with its PDF checkbox.

## Alternate modes

`--agent` explicitly selects the earlier Deep Agents orchestrator.
`--direct` isolates the writer from retrieval and planning. These diagnostic
modes retain factual review and compilation but do not run the new selection-plan
and tailoring-review stages. They are mutually exclusive with `--pipeline`.

## Output and diagnostics

Each run creates a new ignored `outputs/<timestamp>/` directory. `--output-dir`
may select another new directory; existing directories are rejected.

| File | Purpose |
| --- | --- |
| `progress.json` | Current pipeline stage and progress message |
| `retrieval.json` | All requirement searches and selected source IDs |
| `resume_evidence.md` | Retrieved chunks with IDs and metadata |
| `plan_attempt_*.md` | Raw selection-plan responses, including failed attempts |
| `tailoring_plan.json` | Accepted cited plan, decisions, and evidence gaps |
| `attempt_*.md`, `attempt_*.tex` | Diagnostic drafts; these are not accepted exports |
| `response.md`, `trace.json` | Most recent writer response and input/output trace |
| `compile_attempt_*.log` | Preserved compiler log, including errors/timeouts |
| `validation.json` | Running/final status, stage, review and compiler results |
| `optimized_resume.tex` | Exported only after acceptance checks pass |
| `optimized_resume.pdf` | Exported only after actual successful compilation |

`validation.json` is written before long review calls, so a response file is no
longer the only sign of an interrupted run. Keyboard interruption is recorded as
`INTERRUPTED` when Python can handle it; force-killing the process may leave the
last `RUNNING` checkpoint. Earlier diagnostic drafts never count as a successful
run. The UI retains successful downloads across reruns and hides them when inputs
change until you generate a new result.

Exit 0 means the export gate passed. Exit 2 means unresolved validation. Exit 1
means an input/index/model/execution error. Inspect the run's report and compiler
log before rerunning. The job posting is never a source of candidate facts.

## Tests

```powershell
python -m unittest discover -v
$env:RESUME_LIVE_EMBEDDINGS = "1"
python -m unittest tests.test_full_pipeline -v
```

These exercise actual SQLite, Chroma, agent state, validation, UI interactions,
PDF compilation and text extraction. Chat responses are simulated; the second
command additionally uses actual MiniLM weights and tokenizer.

To test live local inference across two different synthetic job descriptions:

```powershell
$env:RESUME_LIVE_CHAT_MODEL = "ollama:qwen3:8b"
python -m unittest tests.test_live_acceptance -v
```

This opt-in test uses real MiniLM, Chroma, your local chat model and pdflatex. It
requires differing relevant project selections, accepted grounding/tailoring
reviews, preserved education dates, and two compiled PDFs. It can take several
minutes. Then run the CLI/UI with your own job and source resume and inspect the
result. Review checks are useful safeguards, not a proof of every factual claim,
layout quality, or ATS performance.
