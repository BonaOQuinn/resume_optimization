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
