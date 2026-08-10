"""
HTML report rendering shared by DataProfiler and DataComparator.

No templating engine dependency (e.g. Jinja2) — plain string
formatting plus pandas' built-in ``to_html`` for the tables, wrapped in
a small self-contained CSS theme.
"""

from __future__ import annotations

import html as _html
from datetime import datetime

import pandas as pd

_CSS = """
:root {
    --bg: #0f1115;
    --panel: #171a21;
    --border: #2a2f3a;
    --text: #e7e9ee;
    --muted: #9aa3b2;
    --accent: #6ea8fe;
}
* { box-sizing: border-box; }
body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 32px;
}
h1 { font-size: 22px; margin-bottom: 4px; }
.meta { color: var(--muted); font-size: 13px; margin-bottom: 28px; }
.section {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 22px;
    overflow-x: auto;
}
.section h2 {
    font-size: 15px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--accent);
    margin: 0 0 12px 0;
}
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td {
    padding: 6px 10px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
}
th { color: var(--muted); font-weight: 600; }
tr:hover td { background: rgba(110, 168, 254, 0.06); }
.empty { color: var(--muted); font-style: italic; }
"""


def _table_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return '<p class="empty">No rows.</p>'
    return df.to_html(index=False, border=0, na_rep="—", classes="report-table")


def render_html_report(
    sections: dict[str, pd.DataFrame],
    title: str = "Data Report",
    subtitle: str | None = None,
) -> str:
    """
    Render a dict of {section_name: DataFrame} into a standalone,
    dark-themed HTML page.
    """

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subtitle_html = f'<div class="meta">{_html.escape(subtitle)}</div>' if subtitle else ""

    body_sections = []
    for name, df in sections.items():
        body_sections.append(
            f'<div class="section"><h2>{_html.escape(str(name))}</h2>'
            f"{_table_html(df)}</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_html.escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>{_html.escape(title)}</h1>
<div class="meta">Generated {generated}</div>
{subtitle_html}
{''.join(body_sections)}
</body>
</html>"""


def save_html_report(
    sections: dict[str, pd.DataFrame],
    path: str,
    title: str = "Data Report",
    subtitle: str | None = None,
) -> str:
    """Render and write the report to ``path``. Returns ``path``."""
    html_text = render_html_report(sections, title=title, subtitle=subtitle)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return path
