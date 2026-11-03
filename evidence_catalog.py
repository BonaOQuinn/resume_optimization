"""Source-owned evidence references and formatting-aware legacy citation checks.

The planner selects IDs. Quotes and selected labels are then copied by the app
from the catalog, never manufactured by the model. No fuzzy/semantic matching
is used to validate legacy quotations or course code/name pairs.
"""

import copy
import json
import re
import unicodedata


COURSE_CODE = r"[A-Za-z]{2,}\s*\d{2,}[A-Za-z]?"


def braced_argument(text, start):
    """Read one balanced TeX argument, allowing nested formatting macros."""
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "{":
        return None, start
    depth, begin = 1, start + 1
    pos = begin
    while pos < len(text):
        if text[pos] == "\\":
            pos += 2
            continue
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if not depth:
                return text[begin:pos], pos + 1
        pos += 1
    return None, start


def readable(text):
    """Remove display markup only, preserving words, negation, numbers and units."""
    text = unicodedata.normalize("NFKC", text)
    # Drop TeX comments, preserving literal escaped percent signs.
    text = re.sub(r"(?<!\\)%[^\n]*", "", text) if "\\" in text else text
    text = re.sub(r"\\(?:begin|end)\{[^}]+\}(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"\\(?:vspace|hspace)\*?\{[^}]*\}", " ", text)
    # A hyperlink's visible text is its second argument.
    while match := re.search(r"\\href\s*\{", text):
        _, end = braced_argument(text, match.end() - 1)
        label, finish = braced_argument(text, end)
        if label is None:
            break
        text = text[:match.start()] + label + text[finish:]
    text = re.sub(r"\\(?:textbf|textit|textsc|textrm|emph|underline|url|section|resumeItem|resumeSubheading|resumeProjectHeading)\*?\b", "", text)
    text = re.sub(r"\\(?:small|Huge|huge|large|scshape|item|resumeSubHeadingListStart|resumeSubHeadingListEnd|resumeItemListStart|resumeItemListEnd)\b", " ", text)
    text = re.sub(r"\\([%&_#$\{\}])", r"\1", text)
    text = text.replace("\\\\", " ").replace("{", " ").replace("}", " ").replace("~", " ")
    text = text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "−": "-"}))
    text = re.sub(r"(?<=\w)\s*--+\s*(?=\w)", "-", text)
    text = re.sub(r"\s+([,.;:%)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def key(text):
    return readable(text).casefold()


def contains_quote(quote, source):
    target = key(quote)
    return len(target) >= 8 and re.search(r"(?<!\w)" + re.escape(target) + r"(?!\w)", key(source)) is not None


def source_content(source):
    return re.sub(r"\A(?:Chunk ID:[^\n]*\n|Metadata:[^\n]*\n)+", "", source).strip()


def metadata(source):
    match = re.search(r"^Metadata:\s*(\{[^\n]*\})", source, re.M)
    if match:
        try:
            value = json.loads(match.group(1))
            return value if isinstance(value, dict) else {}
        except ValueError:
            pass
    return {}


def course_records(source):
    """Read paired code/name fields from one record, or explicit resume entries."""
    records = []
    code = re.search(r"^Course code:\s*(.+)$", source, re.M | re.I)
    name = re.search(r"^Course:\s*(.+)$", source, re.M | re.I)
    meta = metadata(source)
    if name:
        records.append({"code": code.group(1).strip() if code else "", "name": name.group(1).strip()})
    elif meta.get("item_type") == "course" and isinstance(meta.get("name"), str):
        records.append({"code": str(meta.get("code") or ""), "name": meta["name"]})
    # The approved LaTeX resume uses CS577 (Course name). Restrict scanning to
    # coursework if the document has sections, to avoid interpreting job names.
    sections = list(re.finditer(r"\\section\*?\s*\{([^}]+)\}", source))
    course_text = source
    if sections:
        course_text = "\n".join(source[m.end():sections[i + 1].start() if i + 1 < len(sections) else len(source)]
                                for i, m in enumerate(sections) if "course" in m.group(1).casefold())
    plain = readable(course_text)
    for match in re.finditer(r"\b(" + COURSE_CODE + r")\s*\(([^()]+)\)", plain):
        records.append({"code": match.group(1), "name": match.group(2).strip()})
    # Plain Markdown uses one code: name entry per line. Do not flatten lines
    # here or combine a code from one course with the name of the next course.
    for line in course_text.splitlines():
        match = re.match(r"\s*[-*•]?\s*(" + COURSE_CODE + r")\s*:\s*(.+?)\s*$", line)
        if match:
            records.append({"code": match.group(1), "name": readable(match.group(2))})
    return [r for r in records if r["name"]]


def course_matches(label, record):
    """Match a code alone, name alone, or the same record's code/name pair."""
    label = key(label)
    code = re.sub(r"\s+", "", record["code"]).casefold()
    name = key(record["name"])
    if label == name or (code and re.sub(r"\s+", "", label) == code):
        return True
    match = re.fullmatch(r"(" + COURSE_CODE + r")\s*(?::|-|\()\s*(.*?)\)?", label, re.I)
    return bool(match and re.sub(r"\s+", "", match.group(1)).casefold() == code and key(match.group(2)) == name)


def selected_name_supported(name, section, sources):
    if not isinstance(name, str) or not key(name):
        return False
    if section == "courses":
        return any(course_matches(name, record) for source in sources.values() for record in course_records(source))
    return any(re.search(r"(?<!\w)" + re.escape(key(name)) + r"(?!\w)", key(s)) for s in sources.values())


def project_names(source):
    meta = metadata(source)
    if meta.get("item_type") == "project" and meta.get("title"):
        return [meta["title"]]
    if re.search(r"^Type:\s*project\s*$", source, re.M | re.I):
        match = re.search(r"^Title:\s*(.+)$", source, re.M)
        if match:
            return [match.group(1).strip()]
    names = []
    for match in re.finditer(r"\\resumeProjectHeading\b", source):
        argument, _ = braced_argument(source, match.end())
        if argument:
            title = argument.split("$|$", 1)[0].split(r"\textbar", 1)[0]
            if readable(title):
                names.append(readable(title))
    for match in re.finditer(r"^\s*Project:\s*([^,\n]+)", source, re.M | re.I):
        names.append(match.group(1).strip())
    return names


def build_catalog(sources):
    catalog = {"version": 1, "evidence": {}, "projects": {}, "courses": {}}
    seen = {"projects": set(), "courses": set()}
    for source_id, source in sources.items():
        content = source_content(source)
        spans = re.split(r"(?=\\section\b)|(?=\\resume(?:Subheading|ProjectHeading)\b)", content) if source_id == "base_resume" else [content]
        for span in spans:
            raw, text = span.strip(), readable(span)
            if len(text.split()) < 2:
                continue
            eid = f"E{len(catalog['evidence']) + 1:03d}"
            catalog["evidence"][eid] = {"source_id": source_id, "quote": raw, "text": text}
            for section, candidates in (("projects", [{"name": n} for n in project_names(span)]),
                                        ("courses", course_records(span))):
                for candidate in candidates:
                    identity = (key(candidate.get("code", "")), key(candidate["name"]))
                    if identity in seen[section]:
                        continue
                    seen[section].add(identity)
                    rid = ("P" if section == "projects" else "C") + f"{len(catalog[section]) + 1:03d}"
                    catalog[section][rid] = {**candidate, "source_id": source_id, "evidence_id": eid}
    return catalog


def materialize_plan(raw_plan, catalog):
    """Resolve model-selected IDs to source-owned quotes and canonical names.

    Legacy plans with literal quotations remain supported by validate_plan; they
    do not bypass its formatting-aware, non-fuzzy checks.
    """
    plan = copy.deepcopy(raw_plan)
    if not isinstance(plan, dict):
        return plan
    for match in plan.get("requirements", []):
        if not isinstance(match, dict) or "evidence_ids" not in match:
            continue
        ids = match["evidence_ids"]
        if not isinstance(ids, list) or any(not isinstance(i, str) or i not in catalog["evidence"] for i in ids):
            raise ValueError(f"{match.get('id')}: select only evidence IDs listed in evidence_catalog.")
        match["evidence"] = [{"evidence_id": eid, "source_id": catalog["evidence"][eid]["source_id"],
                              "quote": catalog["evidence"][eid]["quote"]} for eid in dict.fromkeys(ids)]
    for section in ("projects", "courses"):
        decision = plan.get(section)
        if not isinstance(decision, dict) or "include_ids" not in decision:
            continue
        ids = decision["include_ids"]
        if not isinstance(ids, list) or any(not isinstance(i, str) or i not in catalog[section] for i in ids):
            raise ValueError(f"{section}: select only IDs listed in the {section} catalog, or an empty list.")
        records = [{"id": rid, **catalog[section][rid]} for rid in dict.fromkeys(ids)]
        decision["selected_records"] = records
        decision["include"] = [(r["code"] + ": " if r.get("code") else "") + r["name"] for r in records]
    return plan
