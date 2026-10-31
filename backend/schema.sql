CREATE TABLE IF NOT EXISTS career_items (
    id INTEGER PRIMARY KEY,
    item_type TEXT NOT NULL
        CHECK (item_type IN ('experience', 'project')),
    title TEXT NOT NULL,
    organization TEXT,
    status TEXT,
    source_file TEXT NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS bullets (
    id INTEGER PRIMARY KEY,
    career_item_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (career_item_id)
        REFERENCES career_items(id)
        ON DELETE CASCADE
);


CREATE TABLE IF NOT EXISTS courses (
    id INTEGER PRIMARY KEY,
    code TEXT,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    source_file TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    source_file TEXT NOT NULL
);