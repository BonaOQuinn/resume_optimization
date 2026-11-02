# Resume Studio

Paste a job description and generate a LaTeX resume using your own experience,
projects, skills, and education. The app retains the approved Jake’s Resume
layout and uses the existing `sentence-transformers/all-MiniLM-L6-v2` embeddings,
SQLite source database, and `resume_minilm_l6_v2_chunks` Chroma collection.

## Start the UI

From this repository in VS Code, with your virtual environment active:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL shown in the terminal (normally http://localhost:8501).

1. Keep your complete source resume at `inputs/private/base_resume.tex`, or
   upload/paste a `.tex`, `.md`, or `.txt` source in the sidebar. Include your
   real contact details, education, dates, roles, and accomplishments.
2. Keep your populated SQLite resume database and MiniLM index available. After
   updating your database, run `python main.py` or use **Refresh index** in the UI.
3. Make sure Ollama is running and the chat model shown in **Generation settings**
   is installed. The default is `ollama:qwen3:8b`; `.env` can set
   `RESUME_CHAT_MODEL` to your existing model. MiniLM remains the embedding model.
4. Paste the **job description** and click **Generate tailored resume**.
5. Download the `.tex` source and, when enabled, the compiled PDF. The match table
   explains which requirements have direct/partial support and which remain gaps.

PDF output requires `pdflatex` and the packages in `templates/jakes_resume.tex`.
The app checks PATH and common Windows MiKTeX/MiKTeX2 directories. If needed, set
`RESUME_PDFLATEX` in `.env` to the full executable path. Uncheck **Generate PDF too**
to generate and review LaTeX without compilation. MiKTeX package auto-install is
disabled during background compilation; install missing packages in MiKTeX Console
using the saved compiler log, then retry. A hidden package prompt cannot complete
an unattended run.

The app binds to your own computer. Source resumes and generated files stay in
ignored local folders. The selected chat provider receives the supplied job and
candidate sources; with a local Ollama model, inference runs locally.

## What changed in tailoring

The pipeline segments the entire posting, including short skill requirements and
requirements near the end. It searches each requirement across experience,
projects, skills, and courses using the same MiniLM index. A bounded set of ranked
chunks enters a selection plan with exact source quotes. The writer uses that
plan to prioritize supported bullets, choose applicable projects, and emphasize
relevant skills. Unsupported qualifications stay in the external gap report.

Acceptance checks cover the approved template, selected factual invariants,
source grounding, actual job relevance, and compilation when requested. An
unchanged substantive resume is rejected. Each draft has at most two correction
attempts. The selection plan also has at most two repair attempts. Both reviews
are model-based and can make mistakes; inspect the downloaded resume before use.

Formatting follows [RESUME_GUIDELINES.md](RESUME_GUIDELINES.md): Education,
Experience, applicable Projects, Technical Skills, then Relevant Courses when
projects provide weak evidence. Empty Projects is omitted. Dates, identity and
education remain source-backed. There is no imposed page-count limit.

See [RUN_AGENT.md](RUN_AGENT.md) for CLI usage and diagnostics and
[VALIDATION.md](VALIDATION.md) for exactly what was tested.
