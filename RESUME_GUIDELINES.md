# Approved resume format: Jake's Resume

Use the Jake's Resume layout supplied in templates/jakes_resume.tex, derived from
the user's uploaded main.tex. It is a formatting reference, not candidate evidence.

## Layout

- Preserve the supplied article class with letterpaper and 11pt, the active font
  configuration, margin adjustments, heading rules, list spacing, links, and
  custom resume commands. Do not enable the commented alternative fonts.
- Use the centered name/contact header and the existing resumeSubheading,
  resumeProjectHeading, and resumeItem commands for the corresponding content.
- Keep candidate information separate from template examples and comments.
  Fill the template with verified source facts; remove all empty slots/comments
  that would leave an unfinished resume.
- Escape special characters in inserted text, including percentages, while
  preserving legitimate LaTeX commands. Hyperlink targets must match the verified
  displayed contact information.
- No page-count limit was specified. Do not assume one page or shrink the
  approved font/margins to force an unstated limit.

## Required section order

After the contact header, retain this relative order:
1. Education
2. Experience
3. Projects
4. Technical Skills
5. Relevant Courses (conditional)

Education is always the first content section. Do not insert Summary, Objective,
or other sections ahead of it. Do not reorder the sections for a particular job.
If an optional section is omitted, the remaining sections retain this order.

## Projects and coursework

- Assess each project's applicability using supported overlap with the target
  job's required/preferred skills, responsibilities, and technologies.
- Include applicable projects; do not fill space with unrelated projects.
- Include Relevant Courses when no applicable project exists, or when the
  available projects are weak evidence and relevant coursework improves job fit.
- When no project is applicable, omit the Projects section completely and include
  relevant, verified courses after Technical Skills. Do not leave an empty
  Projects heading or move Relevant Courses ahead of Technical Skills.
- If applicable projects already provide sufficient evidence, Relevant Courses
  can be omitted. Document the selection decision in the evidence map.
- Retrieve courses explicitly when needed and select only course names/codes
  supported by the source resume/database. Coursework shows academic exposure;
  it does not establish professional experience or an otherwise missing degree.
- If neither projects nor courses are relevant, omit those optional sections
  and report the evidence gap externally rather than inventing content.

## Evidence and return format

Preserve factual education, dates, roles, metrics, and completion status.
Do not treat malformed example values, typos, or placeholders in a template as
facts. Ask when a required source fact is ambiguous.
Return complete LaTeX according to the configured output_mode. Keep internal
evidence references and selection notes outside the resume. Do not claim
compilation, page fit, or PDF generation without a real compilation tool.
