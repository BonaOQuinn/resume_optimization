"""Synchronize the four resume section Markdown files into SQLite."""

import argparse
import re
import sqlite3
from contextlib import closing
from pathlib import Path

try:
    from .init_db import BACKEND_FOLDER, DEFAULT_DATABASE, initialize_database
except ImportError:  # Also support running this file directly.
    from init_db import BACKEND_FOLDER, DEFAULT_DATABASE, initialize_database

SOURCE_FILES = ("experience.md", "projects.md", "relevant_courses.md", "tech_skills.md")
HEADING = re.compile(r"^(#{1,6})\s*(.+?)\s*#*\s*$")
BULLET = re.compile(r"^[-*+]\s+(.+)$")
STATUS = re.compile(r"\s*\((in progress|completed|planned|ongoing)\)\s*$", re.I)


class SourceError(ValueError):
    """A source file does not follow the supported Markdown structure."""


def markdown_blocks(path: Path):
    """Yield headings and bullets, joining wrapped bullet lines with spaces."""
    pending = None
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        heading = HEADING.fullmatch(line)
        bullet = BULLET.fullmatch(line)
        if heading or bullet:
            if pending is not None:
                yield pending
                pending = None
            if heading:
                yield ("heading", len(heading[1]), heading[2].strip(), number)
            else:
                pending = ("bullet", 0, bullet[1].strip(), number)
        elif pending is not None:
            pending = ("bullet", 0, pending[2] + " " + line, pending[3])
        else:
            raise SourceError(f"{path.name}:{number}: expected a heading or bullet")
    if pending is not None:
        yield pending


def require_unique(rows, fields, source):
    keys = [tuple(row[field] for field in fields) for row in rows]
    if len(keys) != len(set(keys)):
        raise SourceError(f"{source}: duplicate entries for {', '.join(fields)}")


def parse_career(path: Path, item_type: str) -> list[dict]:
    items = []
    current = None
    for kind, level, text, number in markdown_blocks(path):
        if kind == "heading":
            if level == 1:
                continue
            if level not in (2, 3):
                raise SourceError(f"{path.name}:{number}: entries must use ## or ###")
            parts = re.split(r"\s+[-–—]\s+", text, maxsplit=1)
            title = parts[0].strip()
            organization = parts[1].strip() if len(parts) == 2 else None
            status = STATUS.search(title)
            if status:
                title = title[:status.start()].strip()
            if not title or organization == "":
                raise SourceError(f"{path.name}:{number}: empty title or organization")
            current = dict(item_type=item_type, title=title, organization=organization,
                           status=status[1].lower() if status else None,
                           source_file=path.name, sort_order=len(items), bullets=[])
            items.append(current)
        elif current is None:
            raise SourceError(f"{path.name}:{number}: bullet has no job or project heading")
        else:
            current["bullets"].append(text)
    if not items or any(not item["bullets"] for item in items):
        raise SourceError(f"{path.name}: expected entries with at least one bullet each")
    require_unique(items, ("title", "organization"), path.name)
    return items


def parse_catalog(path: Path, courses: bool) -> list[dict]:
    rows = []
    category = None
    title_seen = False
    for kind, level, text, number in markdown_blocks(path):
        if kind == "heading":
            if not title_seen:
                title_seen = True
            else:
                category = text
            continue
        if category is None:
            raise SourceError(f"{path.name}:{number}: bullet has no category heading")
        row = dict(category=category, source_file=path.name)
        if courses:
            code, separator, name = text.partition(":")
            if not separator or not code.strip() or not name.strip():
                raise SourceError(f"{path.name}:{number}: use 'COURSE_CODE: Course name'")
            row.update(code=code.strip(), name=name.strip())
        else:
            row["name"] = text
        rows.append(row)
    if not rows:
        raise SourceError(f"{path.name}: expected at least one categorized entry")
    require_unique(rows, ("code",) if courses else ("name", "category"), path.name)
    return rows


def load_sources(source_dir: Path) -> dict:
    """Parse all inputs before opening or modifying a database."""
    return dict(
        experience=parse_career(source_dir / SOURCE_FILES[0], "experience"),
        projects=parse_career(source_dir / SOURCE_FILES[1], "project"),
        courses=parse_catalog(source_dir / SOURCE_FILES[2], courses=True),
        skills=parse_catalog(source_dir / SOURCE_FILES[3], courses=False),
    )


def sync_rows(connection, table, rows, columns, key_columns, scope, scope_values):
    """Update matching rows, insert new rows, and remove stale scoped rows.

    Identifiers are internal constants; all data values use bound parameters.
    IDs for matching identities are preserved for future retrieval references.
    """
    old = {}
    for row in connection.execute(f"SELECT * FROM {table} WHERE {scope}", scope_values):
        key = tuple(row[column] for column in key_columns)
        if key in old:
            raise SourceError(f"{table}: duplicate existing identity {key}; resolve before import")
        old[key] = row["id"]
    ids = []
    for row in rows:
        key = tuple(row[column] for column in key_columns)
        row_id = old.pop(key, None)
        values = tuple(row[column] for column in columns)
        if row_id is None:
            cursor = connection.execute(
                f"INSERT INTO {table} ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})", values)
            row_id = cursor.lastrowid
        else:
            connection.execute(
                f"UPDATE {table} SET {', '.join(column + ' = ?' for column in columns)} WHERE id = ?",
                (*values, row_id))
        ids.append(row_id)
    for row_id in old.values():
        connection.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
    return ids


def populate_database(database: Path, data: dict) -> dict:
    database = initialize_database(database)
    with closing(sqlite3.connect(database)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:  # All four source files are synchronized in one transaction.
            for source, key in zip(SOURCE_FILES[:2], ("experience", "projects")):
                items = data[key]
                ids = sync_rows(connection, "career_items", items,
                    ("item_type", "title", "organization", "status", "source_file", "sort_order"),
                    ("item_type", "title", "organization"), "source_file = ?", (source,))
                for item_id, item in zip(ids, items):
                    bullets = [dict(career_item_id=item_id, text=text, sort_order=position)
                               for position, text in enumerate(item["bullets"])]
                    sync_rows(connection, "bullets", bullets,
                        ("career_item_id", "text", "sort_order"), ("sort_order",),
                        "career_item_id = ?", (item_id,))
            sync_rows(connection, "courses", data["courses"],
                ("code", "name", "category", "source_file"), ("code",),
                "source_file = ?", (SOURCE_FILES[2],))
            sync_rows(connection, "skills", data["skills"],
                ("name", "category", "source_file"), ("name", "category"),
                "source_file = ?", (SOURCE_FILES[3],))
    return counts(data)


def counts(data: dict) -> dict:
    return dict(experience=len(data["experience"]), projects=len(data["projects"]),
                bullets=sum(len(item["bullets"]) for key in ("experience", "projects")
                            for item in data[key]),
                courses=len(data["courses"]), skills=len(data["skills"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=BACKEND_FOLDER.parent)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing")
    args = parser.parse_args()
    try:
        data = load_sources(args.source_dir)
        result = counts(data) if args.dry_run else populate_database(args.database, data)
    except (OSError, SourceError, sqlite3.Error) as error:
        parser.exit(1, f"Import failed: {error}\n")
    print("Validated" if args.dry_run else f"Imported into {args.database.resolve()}")
    print(", ".join(f"{count} {name}" for name, count in result.items()))


if __name__ == "__main__":
    main()
