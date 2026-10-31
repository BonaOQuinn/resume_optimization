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

## Embed and store the resume documents

From the repository root, with your virtual environment active:

```bash
python -m pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in the root `.env` file or in your environment. Then, after
creating and populating `backend/data/resume.db`, run:

```bash
python main.py
```

The command builds LangChain `Document` objects from SQLite, embeds their full
text with `text-embedding-3-large`, and stores text, metadata, IDs, and vectors in
the `resume_collection` Chroma collection at `backend/data/chroma`. Chroma
persists automatically. The default SQLite, Chroma, and `.env` paths are based
on the repository location. Custom paths can be passed with `--database PATH`
and `--persist-directory PATH`.

Importing `main` or running `python main.py --help` does not create an embedding
client or index data. Missing databases fail with a helpful message; empty
databases do not create a vector store.

Running the command again re-embeds every document, so it makes additional API
calls. Matching IDs are updated instead of duplicated. Deleted SQLite records
are not automatically removed from Chroma. Keep the same embedding model when
querying this collection. API credentials and the default generated data paths
are ignored by Git.

Run all checks, including indexing with temporary data and local test embeddings
(no OpenAI API calls):

```bash
python -m unittest discover -s tests -v
```
