"""Create the SQLite tables without inserting resume data."""

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path

BACKEND_FOLDER = Path(__file__).resolve().parent
DEFAULT_DATABASE = BACKEND_FOLDER / "data" / "resume.db"


def initialize_database(database_path: Path = DEFAULT_DATABASE) -> Path:
    database_path = Path(database_path).resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    schema = (BACKEND_FOLDER / "schema.sql").read_text(encoding="utf-8")
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(schema)
    return database_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args()
    print(f"Database initialized at: {initialize_database(args.database)}")


if __name__ == "__main__":
    main()

