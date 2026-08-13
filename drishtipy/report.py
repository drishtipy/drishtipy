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


# =====================================================
# TERMINAL DASHBOARD (used by ProfileReport.__repr__)
# =====================================================

_BAR_LENGTH = 24


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _score_bar(score: float, length: int = _BAR_LENGTH) -> str:
    filled = max(0, min(length, round(score / 100 * length)))
    return "█" * filled + "░" * (length - filled)


def _score_label(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Fair"
    return "Needs Attention"


def _box(lines: list[str], width: int) -> list[str]:
    top = "╭" + "─" * (width - 2) + "╮"
    bottom = "╰" + "─" * (width - 2) + "╯"
    body = [f"│ {line:<{width - 4}} │" for line in lines]
    return [top, *body, bottom]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def render_dashboard(report: "ProfileReport") -> str:
    """Compact, terminal-friendly analytical summary of a ProfileReport."""

    profiler = report._profiler
    df = profiler.df
    n_rows, n_cols = df.shape
    mem_str = _human_bytes(df.memory_usage(deep=True).sum())

    header_lines = [
        report._title,
        f"{n_rows:,} {'row' if n_rows == 1 else 'rows'}  ×  "
        f"{n_cols} {'column' if n_cols == 1 else 'columns'}   ·   {mem_str}",
    ]
    if profiler.is_sampled and profiler.total_rows_in_source is not None:
        header_lines.append(
            f"Sampled: {len(df):,} of {profiler.total_rows_in_source:,} source rows"
        )
    width = max(len(l) for l in header_lines) + 4
    width = max(width, 46)

    lines = [""] + _box(header_lines, width) + [""]

    # ---- quality score ----
    try:
        qs = profiler.quality_score()
        overall = float(qs.loc[qs["Metric"] == "Overall Quality Score", "Score"].iloc[0])
        lines.append(
            f"  Quality Score  {overall:5.1f}/100  {_score_bar(overall)}  "
            f"({_score_label(overall)})"
        )
        dims = qs[qs["Metric"] != "Overall Quality Score"]
        dim_bits = [f"{row.Metric} {row.Score:.0f}" for row in dims.itertuples()]
        lines.append("  " + "   ·   ".join(dim_bits))
    except Exception:
        pass

    lines.append("")

    # ---- alerts ----
    try:
        alerts = profiler.alerts()
        if len(alerts):
            counts = alerts["Severity"].value_counts()
            parts = [
                f"{counts.get(s, 0)} {s}"
                for s in ("High", "Medium", "Low")
                if counts.get(s, 0)
            ]
            lines.append(
                f"  ⚠ {len(alerts)} alert(s): {', '.join(parts)}"
                f"   →  df.profile.alerts()"
            )
        else:
            lines.append("  ✓ No alerts — nothing flagged")
    except Exception:
        pass

    lines.append("")

    # ---- section summary ----
    section_hints = {
        "Schema": lambda d: _plural(len(d), "column"),
        "Statistics": lambda d: f"{_plural(len(d), 'column')} summarized",
        "Quality": lambda d: (
            f"{(d['Missing Count'] > 0).sum()} with missing values, "
            f"{(d['Duplicate Count'] > 0).sum()} with duplicate values"
        ),
        "ML": lambda d: (
            f"{(d['Feature Status'] != 'Good').sum()} flagged for review"
            if "Feature Status" in d.columns
            else _plural(len(d), "column")
        ),
        "ETL": lambda d: (
            f"{(d['ETL Status'] != 'Ready').sum()} need cleaning"
            if "ETL Status" in d.columns
            else _plural(len(d), "column")
        ),
    }
    for name in ("Schema", "Statistics", "Quality", "ML", "ETL"):
        section_df = report.get(name)
        if section_df is None:
            continue
        try:
            hint = section_hints[name](section_df)
        except Exception:
            hint = _plural(len(section_df), "column")
        lines.append(f"  {name:<11}{hint}")

    lines.append("")
    lines.append("  report['Schema'] / ['Statistics'] / ['Quality'] / ['ML'] / ['ETL']")
    lines.append("  Tip: display this in Jupyter for a full interactive HTML report.")
    lines.append("")

    return "\n".join(lines)


class ProfileReport(dict):
    """
    ``dict[str, pandas.DataFrame]`` returned by
    ``DataProfiler.info_dataframe(section="All")`` / ``df.profile.info()``.

    Behaves exactly like a plain dict — ``report["Schema"]``, ``.keys()``,
    ``.values()``, ``.items()``, iteration, ``len()`` all work as before —
    but renders as a full interactive HTML report inline in Jupyter/IPython
    (via ``_repr_html_``), and as a compact analytical dashboard when
    printed in a terminal (via ``__repr__``), instead of a raw dict of
    DataFrames.
    """

    def __init__(self, sections: dict[str, pd.DataFrame], profiler, title: str):
        super().__init__(sections)
        self._profiler = profiler
        self._title = title

    def _repr_html_(self) -> str:
        subtitle = None
        if self._profiler.is_sampled and self._profiler.total_rows_in_source is not None:
            df = self._profiler.df
            subtitle = (
                f"Based on a random sample of {len(df):,} rows "
                f"out of {self._profiler.total_rows_in_source:,} total rows in source."
            )
        return render_html_report(dict(self), title=self._title, subtitle=subtitle)

    def __repr__(self) -> str:
        return render_dashboard(self)
