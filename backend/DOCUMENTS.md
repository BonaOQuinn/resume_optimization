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
runs reuse the download cache. No OpenAI API key, credits, or Hugging Face
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
