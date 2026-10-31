# Prepare a pasted job posting

From the repository root, using Python 3.10 or later:

```bash
python backend/prepare_job_document.py
```

Enter a job title and press Enter. Then paste the description. After its final
line, press Enter and type `.done` on a line of its own. Repeat for required
qualifications and preferred qualifications. Blank lines within pasted text are
allowed. To skip either qualifications field, enter `.done` immediately.
Title and description are required. Closing input before `.done` reports an error.

The resulting JSON object is printed to the terminal. To save it instead:

```bash
python backend/prepare_job_document.py --output backend/data/job_document.json
```

Alternatively, call the function from your Python code:

```python
from backend.prepare_job_document import prepare_job_document

document = prepare_job_document(
    job_title="Machine Learning Engineer",
    description="Build and maintain computer vision services for our users.",
    qualifications="Experience with Python and model evaluation.",
    preferred_qualifications="Experience with FastAPI and AWS.",
)

print(document["text"])
```

This produces labeled sections for the job title, description, required
qualifications, and preferred qualifications. Empty optional sections are omitted.
Common words, complete sentences, paragraphs, bullets, capitalization, and
punctuation such as `C++` or `Node.js` are preserved. Only whitespace at the outer
edges of each field is trimmed. The function performs no keyword extraction,
summarization, model calls, database writes, or embedding generation.

The document has the same `id`, `text`, and `metadata` shape as resume documents.
Its ID is a SHA-256 hash of the prepared fields, prefixed with `job:`. The same
inputs produce the same ID; edited content produces a different ID. Metadata
contains `item_type = job` and the title. Use the `text` value as the eventual
embedding query against your resume documents.

Run the focused checks:

```bash
python -m unittest discover -s tests -p test_prepare_job_document.py -v
```
