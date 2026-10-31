import sqlite3
from pathlib import Path

project_folder = Path(__file__).resolve().parent
database_folder = project_folder / "data"
database_folder.mkdir(exist_ok=True)

database_path = database_folder / "resume.db"
schema_path = project_folder / "schema.sql"

schema = schema_path.read_text(encoding="utf-8")

with sqlite3.connect(database_path) as connection:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(schema)

print(f"Database created at: {database_path}")
