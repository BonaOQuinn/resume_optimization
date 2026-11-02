# Prepare searchable resume documents

Run after populating `backend/data/resume.db`:

```bash
python backend/prepare_documents.py --output backend/data/documents.json
```

Requires Python 3.10+ and the standard library only. Without `--output`, JSON is
printed to the terminal. `--database PATH` selects another SQLite database.
The default database path works regardless of the terminal's current folder.

The JSON array contains one document for each job, project, course, and skill.
Each document has `id`, `text`, and `metadata`. Career documents contain the
title, optional organization/status, and every accomplishment in database order.
Course documents include code (if present), name, and category. Skill documents
include name and category. Missing information is omitted rather than invented.
Records with no bullets still produce a document from their available fields;
an empty table produces an empty list for that type.

All original sentences, common words, capitalization, Unicode, and technical
punctuation such as C++ and Node.js are preserved. Preparation only adds labels
and bullet markers; it does not summarize, split skill names, remove stop words,
or extract keywords. Any earlier whitespace joining happened during import.

Use the `text` field as input to an embedding model. Store `id` and `metadata`
alongside its embedding to map results back to the source database. IDs use the
prefixes `career_item:`, `course:`, and `skill:` plus each database primary key.
They stay stable while the underlying record's ID stays stable. Text changes
require re-embedding, and deleted records must also be removed from a future index.

Functions can also be called directly:

```python
from backend.prepare_documents import prepare_documents

documents = prepare_documents()  # Read-only access to the default database.
for document in documents:
    print(document["id"], document["text"])
```

For an existing SQLite connection, call `prepare_career_documents(connection)`,
`prepare_course_documents(connection)`, `prepare_skill_documents(connection)`,
or `prepare_all_documents(connection)`. Career preparation optionally accepts
`item_type="experience"` or `item_type="project"`. These functions do not close
the caller's connection or change its row factory. The path-based function reads
all sections in one transaction and never creates a missing database.

Run the document preparation checks:

```bash
python -m unittest discover -s tests -p test_prepare_documents.py -v
```

## Split, embed, and store the resume documents

From the repository root, with your virtual environment active:

```bash
python -m pip install -r requirements.txt
python main.py
```

The SQLite database at `backend/data/resume.db` must already be populated.
`--database PATH` selects another database, and `--persist-directory PATH`
selects another Chroma storage directory.

The pipeline now runs in this order:

1. Read jobs, projects, courses, and skills from SQLite.
2. Convert the prepared records into LangChain `Document` objects.
3. Split longer documents with `RecursiveCharacterTextSplitter`, measuring
   length with the MiniLM tokenizer rather than character counts.
4. Embed each chunk locally with `sentence-transformers/all-MiniLM-L6-v2`
   on the CPU, with normalized output vectors.
5. Save the chunks to the new `resume_minilm_l6_v2_chunks` collection inside
   `backend/data/chroma`.

The first run downloads the public model and tokenizer from Hugging Face; later
runs reuse the project cache at `backend/data/huggingface`. Both the splitter's
tokenizer and the embedding model use its `hub` subfolder; Xet data also stays
inside the project cache. No OpenAI API key, credits, or Hugging Face
inference subscription is required for this local embedding path. The optional
root `.env` is still loaded for settings such as LangSmith tracing.

The model produces 384-dimensional vectors and has about 22.7 million parameters.
Its default input limit is 256 word pieces. Chunks target 220 content tokens
with up to 30 tokens of overlap, preferring paragraph, line, and word boundaries.
Short documents usually remain a single chunk. Every final chunk is checked
including special tokens against the 256-token limit before embedding, so an
oversized chunk fails instead of being silently truncated. Sentences and common
words remain in the text; no keyword extraction or stemming is applied.

Each chunk keeps its original metadata and adds `parent_id`, `chunk_index`,
`chunk_count`, and `token_count` (including model special tokens). Chunk IDs
look like `career_item:3:chunk:0`. The parent ID maps back to the source SQLite
record. `split_resume_documents(documents)` in `backend/chunk_documents.py` can
also be used directly with an existing list of LangChain documents.

The existing `resume_collection` is preserved. Do not mix vectors from different
embedding models in one collection, even if their dimensions happen to match.
Query the new collection with the same MiniLM model and normalization settings;
`create_vector_store()` in `main.py` opens it with those settings.

Each run recomputes embeddings locally and updates matching chunk IDs. Surplus
chunks for shortened source documents are removed only after their replacement
chunks are saved successfully. Parents deleted entirely from SQLite are not
automatically removed. Empty source databases leave any existing store unchanged.
There is no external embedding API charge for rerunning, but it uses local CPU.

Importing `main` or requesting `python main.py --help` does not download a model
or index data. Generated SQLite/Chroma data and API credentials are ignored by Git.

Run the checks with an in-memory tokenizer, temporary databases, and local test
embeddings (no model downloads or API calls):

```bash
python -m unittest discover -s tests -v
```

Model details: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## Windows model-cache permissions

If a previous run failed with `WinError 5` or `PermissionError` under
`C:/Users/.../.cache/huggingface`, pull the latest code and restart
`python main.py`. The application configures its project cache before importing
Hugging Face libraries, because those libraries read cache settings at import
time. It also passes the same cache folder explicitly to both model loaders.
The inaccessible global cache does not need to be deleted or modified.

To choose another writable folder, add this setting to your root `.env`:

```dotenv
RESUME_MODEL_CACHE=C:/Users/your-name/resume-model-cache
```

Relative paths are resolved from the repository root. Restart Python after
changing the setting. The default `backend/data/huggingface/` folder is ignored
by Git. A warning about unauthenticated Hugging Face requests is separate from
cache permissions; this public model can download without a token.

## Resume retrieval tool for Deep Agents

Install the updated dependencies with `python -m pip install -r requirements.txt`.
The root `agent.py` defines `search_resume` and its shared `backend`. It opens
the existing MiniLM collection lazily on the first search and retrieves up to
four chunks per query. It does not re-index or re-embed the stored documents;
only the search query is embedded.

The tool writes each match under `/retrieved/resume/{batch_id}/chunk_N.md` with
its full chunk text, source filename, chunk ID, and original metadata. That
includes the parent SQLite identity, item type, and available title, organization,
course, or skill information. Consecutive searches use different batch IDs.
Empty queries and empty collections return helpful messages; failed uploads
are reported separately from successfully saved paths.

These paths belong to the agent's virtual filesystem in LangGraph state, not
your computer's project folders. Run the tool through a Deep Agent using the
same backend instance so built-in tools such as `read_file` can read the files.
A standalone `search_resume.invoke(...)` outside a graph cannot write to
`StateBackend`.

The resume-generation helpers below connect this retrieval tool and shared
backend to the orchestrator and its specialist prompts.

MiniLM handles embedding and retrieval. The agent's chat model is a separate
choice; this module does not select a provider or make chat model API calls.
Running `python agent.py` only loads definitions. The retrieval tests exercise
saving and reading files in a real graph without a chat model:

```bash
python -m unittest tests.test_agent -v
```

## Resume refinement prompts and workflow

`prompts.py` defines the orchestrator workflow and three subagent roles:

| Role | Responsibility |
| --- | --- |
| Orchestrator | Analyze the job, retrieve evidence, select supported content, draft the complete LaTeX resume, and apply review corrections. |
| `job-requirements-analyst` | Separate explicit required/preferred qualifications and propose focused retrieval queries. |
| `resume-evidence-analyst` | Map supplied evidence files to requirements, retaining source references and reporting gaps. |
| `latex-resume-reviewer` | Check the draft against source facts, the exact guidelines/template, and obvious LaTeX source errors. |

Subagents receive explicit task/file paths. They do not automatically inherit
the orchestrator's conversation. They use built-in filesystem tools; the custom
`search_resume` tool stays with the orchestrator. The orchestrator and reviewer
both receive the user's formatting rules and optional LaTeX template.

No font, margins, page count, section order, or template is chosen by default.
`create_resume_agent` requires nonempty `latex_guidelines`; it cannot create a
fully configured agent until you provide them. Useful guidelines include page
limit, paper size, fonts, margins, section order, bullet style, date/link formats,
allowed packages/compiler, content constraints, and any exact template macros.

Return modes:
- `latex_only`: the completed response is raw, complete LaTeX source without
  Markdown fences or commentary.
- `latex_with_notes` (default): one complete LaTeX code block plus brief changes
  and limitations outside the source.

Both modes ask a clarification question if a field required by the user's
template or a fact needed for a selected claim is missing. An unmet job
qualification is reported as a gap and does not automatically block generation.
The workflow preserves ongoing/planned status and exact documented metrics.
It does not infer qualifications from similarity alone.

Once you have initialized your chosen chat model and prepared the input text:

```python
from agent import create_resume_agent, prepare_resume_request

resume_agent = create_resume_agent(
    model=your_chat_model,
    latex_guidelines=your_guidelines_text,
    latex_template=your_template_text,  # Optional; omit if no template.
    output_mode="latex_only",
)
request = prepare_resume_request(
    job_document=your_prepared_job["text"],
    base_resume=your_full_base_resume_text,
)
result = resume_agent.invoke(request)
print(result["messages"][-1].content)
```

`your_prepared_job` can come from `prepare_job_document` in
`backend/prepare_job_document.py`. The source resume should include all facts
needed for the finished document. The current `resume_sections/resume.md`
omits contact details, education, and employment dates; supply those facts
before expecting a finished resume that requires them.

`prepare_resume_request` attaches the provided source texts as
`/inputs/job.md` and `/inputs/base_resume.md` in agent state.
The orchestrator records its evidence map at `/work/resume_evidence.md` and
the final source at `/output/optimized_resume.tex`. These are virtual paths;
the calling application can export the final file from `result["files"]`.

The prompts require factual grounding, preserve valid template commands while
escaping inserted text, and request review before delivery. These are model
instructions, not a deterministic guarantee. No LaTeX compiler or PDF exporter
is connected yet, so the agent must not claim compilation, verified page fit,
or PDF creation. A chosen chat model and real user guidelines are still required
to generate an actual optimized resume.
