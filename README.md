# Resume optimization

## Populate SQLite from Markdown

Use Python 3.10 or later. These scripts use only the standard library;
no LangChain, API keys, or pip packages are required.

From the repository root:

```bash
python backend/populate_db.py --dry-run
python backend/populate_db.py
```

The second command initializes missing tables using `backend/schema.sql`, then
imports the section files into `backend/data/resume.db`:

| Source | Destination |
| --- | --- |
| `experience.md` | `career_items` (`item_type = experience`) and `bullets` |
| `projects.md` | `career_items` (`item_type = project`) and `bullets` |
| `relevant_courses.md` | `courses` |
| `tech_skills.md` | `skills` |

`resume.md` repeats the section files and is excluded to avoid duplication.
Default paths are anchored to the script location, so running from `backend/`
with `python populate_db.py` also works. To initialize empty tables only, run
`python backend/init_db.py`.

To choose another database or source folder:

```bash
python backend/populate_db.py --database backend/data/example.db --source-dir .
```

### Markdown format

Experience and project files start with a section title, followed by `##` or
`###` entry headings and achievement bullets. A heading such as
`### Handles(in progress) - personal` becomes title `Handles`, status
`in progress`, and organization `personal`. Recognized status suffixes are
`in progress`, `completed`, `planned`, and `ongoing`. Other parenthetical text
is retained. Missing organization and status fields stay NULL.

Course and skill files start with a document title, followed by category
headings and bullets. Courses use `CS549: Machine Learning`. Each skill bullet
is one record, including combined text such as `linux / openbsd / azure /`.

Both existing headings like `#Experience` and standard headings like
`# Experience` are accepted. Wrapped bullet lines are joined with spaces.
Wording, capitalization, and source typos are preserved.
Missing files, empty sections, duplicate identities, and unsupported structure
fail validation before data changes. Use `--dry-run` to check edits first.

### Rerunning after edits

Edit the Markdown and run the importer again. Matching records are updated in
place, new records are inserted, and records removed from a section file are
removed from that file's imported records. Jobs/projects match by type, title,
and organization; courses by code; skills by name and category. Bullet IDs
track their position within an item. Renaming an identity creates a new record.
Only rows attributed to the four listed source files are synchronized.
All four sections are written in one transaction; a failure rolls back the
data changes. Schema initialization can create empty tables beforehand.

The existing tracked `backend/data/resume.db` is not replaced in Git. Run the
importer locally to populate it. `.gitignore` excludes new database files but
does not untrack the existing file; avoid committing local database changes.

### Checks

```bash
python -m unittest discover -s tests -v
```
