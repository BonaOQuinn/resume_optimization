"""Combine pasted job details into a full-text document for future embedding."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import TypedDict


class JobDocument(TypedDict):
    id: str
    text: str
    metadata: dict[str, str]


def prepare_job_document(
    job_title: str,
    description: str,
    qualifications: str = "",
    preferred_qualifications: str = "",
) -> JobDocument:
    """Return one labeled job document, preserving full sentences and formatting.

    Title and description are required. Qualifications fields can be empty.
    Only whitespace at the start/end of each field is removed; internal words,
    paragraphs, bullets, capitalization, and technical punctuation are retained.
    The ID is deterministic for the same prepared fields and changes on edits.
    """
    inputs = {
        "job_title": job_title,
        "description": description,
        "qualifications": qualifications,
        "preferred_qualifications": preferred_qualifications,
    }
    fields = {}
    for name, value in inputs.items():
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        fields[name] = value.strip()
    for name in ("job_title", "description"):
        if not fields[name]:
            raise ValueError(f"{name} cannot be empty")

    sections = [
        f"Job title: {fields['job_title']}",
        f"Description:\n{fields['description']}",
    ]
    if fields["qualifications"]:
        sections.append(f"Required qualifications:\n{fields['qualifications']}")
    if fields["preferred_qualifications"]:
        sections.append(f"Preferred qualifications:\n{fields['preferred_qualifications']}")

    digest = hashlib.sha256(
        json.dumps(fields, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "id": f"job:{digest}",
        "text": "\n\n".join(sections),
        "metadata": {"item_type": "job", "title": fields["job_title"]},
    }


def read_pasted_field(label: str) -> str:
    """Read multiline terminal input; a line containing only .done ends a field."""
    print(f"\n{label} (paste text; finish with .done on its own line):", file=sys.stderr)
    lines = []
    while True:
        try:
            line = input()
        except EOFError as error:
            raise ValueError(f"Input ended before .done for {label}") from error
        if line == ".done":
            return "\n".join(lines)
        lines.append(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON file; defaults to stdout")
    args = parser.parse_args()
    try:
        print("Job title:", file=sys.stderr)
        try:
            title = input()
        except EOFError as error:
            raise ValueError("Input ended before the job title") from error
        document = prepare_job_document(
            job_title=title,
            description=read_pasted_field("Description"),
            qualifications=read_pasted_field("Required qualifications (optional)"),
            preferred_qualifications=read_pasted_field("Preferred qualifications (optional)"),
        )
        payload = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
            print(f"Saved job document to {args.output}", file=sys.stderr)
        else:
            print(payload, end="")
    except (TypeError, ValueError, OSError) as error:
        parser.exit(1, f"Job document preparation failed: {error}\n")
    except KeyboardInterrupt:
        parser.exit(1, "\nJob document preparation cancelled.\n")


if __name__ == "__main__":
    main()
