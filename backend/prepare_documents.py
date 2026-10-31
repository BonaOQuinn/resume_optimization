"""Prepare embedding-ready text documents from the structured resume database.

No tokenization, keyword extraction, stop-word removal, or AI calls are used.
Database text is preserved, including sentences, capitalization, and punctuation.
"""

import argparse
import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from typing import TypedDict


DEFAULT_DATABASE = Path(__file__).resolve().parent / "data" / "resume.db"


class Document(TypedDict):
    id: str
    text: str
    metadata: dict[str, str | int | None]


def prepare_career_documents(
    connection: sqlite3.Connection, item_type: str | None = None
) -> list[Document]:
    """Return one document per job/project, with all of its ordered bullets.

    Pass 'experience' or 'project' to select a single type, or omit the argument
    to include both. The caller owns the connection; no row factory is changed.
    """
    if item_type not in (None, "experience", "project"):
        raise ValueError("item_type must be 'experience', 'project', or None")

    bullets = defaultdict(list)
    for parent_id, text in connection.execute(
        "SELECT career_item_id, text FROM bullets ORDER BY career_item_id, sort_order, id"
    ):
        bullets[parent_id].append(text)

    query = """SELECT id, item_type, title, organization, status, source_file
               FROM career_items"""
    parameters = ()
    if item_type is not None:
        query += " WHERE item_type = ?"
        parameters = (item_type,)
    query += " ORDER BY item_type, sort_order, id"

    documents = []
    for record_id, kind, title, organization, status, source_file in connection.execute(
        query, parameters
    ):
        lines = [f"Type: {kind}", f"Title: {title}"]
        if organization:
            lines.append(f"Organization: {organization}")
        if status:
            lines.append(f"Status: {status}")
        if bullets[record_id]:
            lines.extend(["", "Accomplishments:"])
            lines.extend(f"- {text}" for text in bullets[record_id])

        documents.append({
            "id": f"career_item:{record_id}",
            "text": "\n".join(lines),
            "metadata": {
                "record_id": record_id,
                "source_table": "career_items",
                "source_file": source_file,
                "item_type": kind,
                "title": title,
                "organization": organization,
                "status": status,
            },
        })
    return documents


def prepare_course_documents(connection: sqlite3.Connection) -> list[Document]:
    """Return one document per course, preserving its code, name, and category."""
    documents = []
    for record_id, code, name, category, source_file in connection.execute(
        "SELECT id, code, name, category, source_file FROM courses ORDER BY category, id"
    ):
        lines = ["Type: course"]
        if code:
            lines.append(f"Course code: {code}")
        lines.extend([f"Course: {name}", f"Category: {category}"])
        documents.append({
            "id": f"course:{record_id}",
            "text": "\n".join(lines),
            "metadata": {
                "record_id": record_id,
                "source_table": "courses",
                "source_file": source_file,
                "item_type": "course",
                "code": code,
                "name": name,
                "category": category,
            },
        })
    return documents


def prepare_skill_documents(connection: sqlite3.Connection) -> list[Document]:
    """Return one document per skill, without splitting or rewriting skill text."""
    return [
        {
            "id": f"skill:{record_id}",
            "text": f"Type: skill\nSkill: {name}\nCategory: {category}",
            "metadata": {
                "record_id": record_id,
                "source_table": "skills",
                "source_file": source_file,
                "item_type": "skill",
                "name": name,
                "category": category,
            },
        }
        for record_id, name, category, source_file in connection.execute(
            "SELECT id, name, category, source_file FROM skills ORDER BY category, id"
        )
    ]


def prepare_all_documents(connection: sqlite3.Connection) -> list[Document]:
    """Combine career, course, and skill documents with distinct ID prefixes."""
    return (
        prepare_career_documents(connection)
        + prepare_course_documents(connection)
        + prepare_skill_documents(connection)
    )


def prepare_documents(database_path: Path = DEFAULT_DATABASE) -> list[Document]:
    """Read an existing database in one consistent snapshot without modifying it."""
    database_path = Path(database_path).resolve()
    # Read-only mode also prevents SQLite from creating a missing database.
    with closing(sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        return prepare_all_documents(connection)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, help="Optional JSON file; defaults to stdout")
    args = parser.parse_args()
    try:
        if args.output:
            same_path = args.output.resolve() == args.database.resolve()
            same_file = (args.output.exists() and args.database.exists()
                         and args.output.samefile(args.database))
            if same_path or same_file:
                raise ValueError("Output must be different from the SQLite database")
        documents = prepare_documents(args.database)
        payload = json.dumps(documents, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
    except (OSError, sqlite3.Error, ValueError) as error:
        parser.exit(1, f"Document preparation failed: {error}\n")


if __name__ == "__main__":
    main()
