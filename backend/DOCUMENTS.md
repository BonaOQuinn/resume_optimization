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
four chunks per query. The optional `item_type` argument filters to `experience`,
`project`, `course`, or `skill`; use `course` to retrieve coursework explicitly.
It does not re-index or re-embed the stored documents;
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
choice configured with `RESUME_CHAT_MODEL`; chat inference happens when the
created agent is invoked.
Running `python agent.py` initializes the configured chat model and creates the
agent; it does not invoke it or generate a resume. Importing `agent.py` remains
lazy and does not initialize a chat model. The retrieval tests exercise
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

The approved defaults are in `RESUME_GUIDELINES.md` and
`templates/jakes_resume.tex`, based on the uploaded Jake's Resume example.
They preserve its letterpaper/11pt article layout, active font setup, margins,
headings, and custom commands. No page-count limit has been specified.

After the contact header, sections follow Education, Experience, Projects,
Technical Skills, and Relevant Courses. Education always comes first. Projects
and Relevant Courses are conditional: omit unrelated projects, and include
verified coursework when the available projects are unrelated or weak and
coursework improves the match. Relevant Courses remains after Technical Skills.
Omit optional sections entirely instead of leaving empty headings.

`create_resume_agent(model=your_chat_model)` loads these defaults automatically.
Optional `latex_guidelines` add further constraints; the approved layout and
section order remain in both the orchestrator and reviewer prompts. Conflicting
rules trigger a clarification rather than silently changing the format.
An optional `latex_template` should be a compatible Jake-style variant.

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

Once your chat model is configured and the input text is prepared:

```python
from agent import create_resume_agent, prepare_resume_request

resume_agent = create_resume_agent(
    # model=your_chat_model,  # Optional pre-initialized model override.
    # Optional additional constraints or a compatible Jake-style template:
    # latex_guidelines=your_additional_rules,
    # latex_template=your_template_text,
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
or PDF creation. A chosen chat model and complete source facts are still required
to generate an actual optimized resume.

The reusable template includes the original Jake template attribution and
formatting macros, with empty content slots for verified candidate information.
It does not supply candidate facts. Your original uploaded LaTeX can be provided
as `base_resume` if it is your current factual resume; the agent still flags
ambiguous dates/metrics instead of inferring corrections.

## Initialize the chat model and create the agent

Install the updated dependencies, then select a tool-calling chat model in your
existing root `.env` using the `RESUME_CHAT_MODEL` setting. `.env.example` lists
the setting without real credentials. No chat model/provider is hardcoded.

The value must use `provider:model` syntax. For a locally installed Ollama model,
replace the placeholder with the exact tag shown by `ollama list`:

```dotenv
RESUME_CHAT_MODEL=ollama:YOUR_INSTALLED_MODEL
```

The `langchain-ollama` client integration is included in requirements. The Ollama
application/service and a downloaded model with tool-calling support are needed
for local inference; installing the Python package alone does not install them.
Hosted providers need their own LangChain integration package and credentials.

```bash
python -m pip install -r requirements.txt
python agent.py
```

An explicit model flag overrides the environment setting:

```bash
python agent.py --model ollama:YOUR_INSTALLED_MODEL --output-mode latex_only
```

The entry point explicitly runs `initialize_chat_model(...)`, then
`create_resume_agent(model=model, ...)`, and prints when the client and graph
are constructed. That confirms construction, not a live model response or
server availability. It makes no inference request and does not download an
Ollama model or re-index the resume collection.

Programmatic creation can use either form:

```python
from agent import initialize_chat_model, create_resume_agent, prepare_resume_request

# Select the model from RESUME_CHAT_MODEL in .env:
resume_agent = create_resume_agent()

# Or explicitly initialize a model first, as in the LangChain tutorial:
model = initialize_chat_model("ollama:YOUR_INSTALLED_MODEL")
resume_agent = create_resume_agent(model=model, output_mode="latex_only")

# Invoke only when actual job and full resume source text are ready:
result = resume_agent.invoke(prepare_resume_request(job_text, base_resume_text))
```

MiniLM remains the separate local embedding model. The selected chat model
powers the orchestrator and subagents, including tool use and LaTeX generation.
