"""Synthetic candidate facts for reproducible generation and compilation tests."""

from prompts import DEFAULT_TEMPLATE_PATH

SOURCE = """Demo Candidate; email demo@candidate.test.
Education: Bachelor of Science in CS, Example University, Demo City, 2025--2027.
Experience: Engineer, Example Company, Demo City, 2025--Present.
Built Python and SQL services; increased bookings by 18%.
Project: Python API, 2026. Built an API in Python.
Technical Skills: Python, SQL.
"""
BODY = r"""
\begin{center}
\textbf{\Huge Demo Candidate} \\
\href{mailto:demo@candidate.test}{demo@candidate.test}
\end{center}
\section{Education}
\resumeSubHeadingListStart
\resumeSubheading{Example University}{2025--2027}{Bachelor of Science in CS}{Demo City}
\resumeSubHeadingListEnd
\section{Experience}
\resumeSubHeadingListStart
\resumeSubheading{Engineer}{2025--Present}{Example Company}{Demo City}
\resumeItemListStart
\resumeItem{Built Python and SQL services; increased bookings by 18\%.}
\resumeItemListEnd
\resumeSubHeadingListEnd
\section{Projects}
\resumeSubHeadingListStart
\resumeProjectHeading{Python API}{2026}
\resumeItemListStart
\resumeItem{Built an API in Python.}
\resumeItemListEnd
\resumeSubHeadingListEnd
\section{Technical Skills}
\textbf{Languages}: Python, SQL
"""
LATEX = DEFAULT_TEMPLATE_PATH.read_text(encoding="utf-8").partition(r"\begin{document}")[0] + "\\begin{document}\n" + BODY + "\n\\end{document}"
PASS = '{"verdict":"PASS","issues":[]}'
