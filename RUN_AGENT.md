# Run the resume agent for PlayStation

Run these commands on the Windows computer running Ollama, from the project
root with your virtual environment active. `agent.py` initializes the agent;
`run_resume_agent.py` invokes it with your job and resume inputs.

1. Install dependencies: `python -m pip install -r requirements.txt`.
2. Download the chat model if needed: `ollama pull qwen3:8b`.
   Keep the PowerShell window running `ollama serve` open if using the manual
   server setup. The model must appear in `ollama list`.
3. Place your complete original LaTeX resume at
   `inputs/private/base_resume.tex`. This directory is ignored by Git.
   Include identity/contact, education, employment dates, and source facts.
   The tracked Markdown sections alone do not contain all of these fields.
4. Ensure the resume vector collection is populated with `python main.py`.
5. Invoke the agent:

```powershell
python run_resume_agent.py --model ollama:qwen3:8b --job-file jobs/playstation.md --resume-file inputs/private/base_resume.tex
```

Alternatively, set `RESUME_CHAT_MODEL=ollama:qwen3:8b` in `.env` and omit
`--model`. The PlayStation text is the posting provided by the user; its exact
job title was not supplied.

Each run creates a new ignored `outputs/<timestamp>/` directory containing
`response.md`, and, when written by the agent, `optimized_resume.tex` and
`resume_evidence.md`. Read the response for review findings or clarifications.
If the agent asks a question and writes no LaTeX, the runner saves the question
and exits with status 2. Missing inputs, connection errors, empty indexes, or
execution errors produce status 1. An existing output directory is never reused.

The saved LaTeX is a draft. The runner does not compile it or guarantee that
an agent review passed. It exports only the known output paths and keeps your
original input files untouched. Inference uses the configured provider; an
Ollama localhost server on your computer is not accessible from ChatGPT's
hosted workspace.

## Test LaTeX generation without delegation

If the agent offers to review a resume or asks for inputs that were already
supplied, isolate generation with the same complete source files:

```powershell
python run_resume_agent.py --direct --model ollama:qwen3:8b --job-file jobs/playstation.md --resume-file inputs/private/base_resume.tex
```

`--direct` sends the complete source resume, job posting, approved formatting
rules, and template to the chat model in one request. It skips Chroma retrieval,
filesystem tools, and subagents, so a populated index is not required for this
test. It is a generation smoke test rather than a test of the full RAG workflow.
Omit `--direct` to run the existing agent workflow.

The runner saves `response.md` in a fresh `outputs/<timestamp>/` folder. It also
saves `optimized_resume.tex` when the response contains a complete LaTeX
document. One optional LaTeX code fence and separate completed thinking blocks
are handled automatically. Clarifications, generic offers to help, and truncated
documents produce exit status 2 without a `.tex` draft. This boundary check does
not prove factual accuracy or compilation; review and compile the draft before
using it. No retrieval evidence map or specialist review is produced in direct
mode. If direct mode succeeds while agent mode fails, generation works for that
input and the delegated workflow needs further investigation.

