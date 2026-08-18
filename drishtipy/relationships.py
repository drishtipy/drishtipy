"""
Relationship Discovery Engine.

Automatically discovers, scores, ranks, and explains relationships
between columns in a DataFrame, instead of requiring the user to
manually pick and test column pairs one at a time. For N columns
there are N*(N-1)/2 possible pairs; this module filters out the
pairs that can't be meaningfully analyzed (ID columns, constants,
free text, ...), picks a statistically appropriate method for each
of the remaining pairs based on their *semantic* types (not just
pandas dtype), and ranks what's left by effect size.

Design notes / honest scope
----------------------------
- **scipy is optional.** Effect sizes (Pearson/Spearman correlation,
  eta-squared, Cramér's V) are computed with pandas/numpy alone and
  always available. P-values (and therefore FDR-corrected
  significance) require scipy — if it isn't installed, this module
  still runs, but the ``Significance``/``P-Value`` columns report
  ``"Unknown (scipy not installed)"`` instead of silently guessing.
  Install it with ``pip install drishtipy[relationships]``.
- **``.graph()`` returns a data structure, not a rendered image.**
  Building an actual force-directed layout needs a graph/plotting
  stack (networkx, matplotlib) that would contradict the
  pandas-only philosophy of this library. Instead it returns nodes
  and weighted edges, a readable adjacency-list ``__repr__``/HTML
  view, and is trivial to hand to networkx/plotly yourself if you
  want a rendered diagram.
- **Mutual Information** is computed for categorical<->categorical
  pairs only (normalized 0-1, alongside Cramér's V) — extending it
  to every pair type via binning was cut to keep this module's scope
  finite; Cramér's V/eta-squared already cover those cases well.
"""

from __future__ import annotations

import html as _html
import itertools
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd

from .semantic import detect_semantic_type
from .report import render_html_report

try:
    from scipy import stats as _scipy_stats

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    _HAS_SCIPY = False

_NUMERIC_LIKE = {"numeric", "currency", "percentage", "latitude", "longitude"}
_CATEGORICAL_LIKE = {"categorical", "boolean"}
_DATE_LIKE = {"date", "datetime"}
_EXCLUDED = {"id", "text", "constant", "email", "phone"}

_DEPENDENCY_SOURCE_TYPES = {"id", "categorical", "boolean"}
_DEPENDENCY_TARGET_TYPES = {"id", "categorical", "boolean", "text", "date", "datetime"}


# =====================================================
# CONFIG
# =====================================================


@dataclass
class RelationshipConfig:
    """
    Tunable thresholds and limits for :class:`RelationshipAnalyzer`.
    Every field has a sensible default — only override what you need.
    """

    # Classification thresholds (on a 0-1 strength scale).
    strong_threshold: float = 0.80
    moderate_threshold: float = 0.50
    weak_threshold: float = 0.20  # below this -> "Not Meaningful"

    # Statistical validity.
    min_sample_size: int = 30
    alpha: float = 0.05
    fdr_correction: bool = True

    # Pair filtering.
    max_categories: int = 50  # categorical columns above this are skipped (too many groups)
    id_uniqueness_threshold: float = 0.95
    include_id_pairs: bool = False  # advanced override — analyze ID columns anyway
    use_kendall: bool = False  # Kendall is O(n^2); opt-in only

    # Performance / scale.
    sample_size: int | None = 100_000  # rows; None disables sampling
    random_state: int = 42
    max_pairs: int = 5_000  # safety cap on how many pairs get analyzed

    # Dependency / redundancy detection.
    functional_dependency_threshold: float = 0.95
    redundancy_threshold: float = 0.95


# =====================================================
# STATISTICAL HELPERS (numpy/pandas only; scipy optional)
# =====================================================


def _manual_chi2(table: pd.DataFrame) -> float:
    observed = table.values.astype(float)
    total = observed.sum()
    if total == 0:
        return 0.0
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / total
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0)
    return float(terms.sum())


def _cramers_v(sub: pd.DataFrame, col_a: str, col_b: str):
    table = pd.crosstab(sub[col_a], sub[col_b])
    n = table.values.sum()
    if n == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0, None, table

    p_value = None
    if _HAS_SCIPY:
        try:
            chi2, p_value, _dof, _exp = _scipy_stats.chi2_contingency(table)
        except Exception:
            chi2 = _manual_chi2(table)
    else:
        chi2 = _manual_chi2(table)

    r, k = table.shape
    denom = n * (min(r, k) - 1)
    v = float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0
    return min(v, 1.0), p_value, table


def _normalized_mutual_info(table: pd.DataFrame) -> float:
    observed = table.values.astype(float)
    n = observed.sum()
    if n == 0:
        return 0.0
    pxy = observed / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        outer = px @ py
        ratio = np.where((pxy > 0) & (outer > 0), pxy / outer, 1.0)
        mi = np.sum(np.where(pxy > 0, pxy * np.log(ratio), 0.0))
    hx = -np.sum(np.where(px > 0, px * np.log(px), 0.0))
    hy = -np.sum(np.where(py > 0, py * np.log(py), 0.0))
    denom = min(float(hx), float(hy))
    if denom <= 0:
        return 0.0
    return float(min(max(mi / denom, 0.0), 1.0))


def _eta_squared(sub: pd.DataFrame, cat_col: str, num_col: str, max_categories: int):
    groups = sub.groupby(cat_col, observed=True)[num_col]
    if groups.ngroups < 2 or groups.ngroups > max_categories:
        return None

    grand_mean = sub[num_col].mean()
    ss_total = float(((sub[num_col] - grand_mean) ** 2).sum())
    if ss_total == 0:
        return 0.0, None, groups.ngroups

    group_means = groups.mean()
    group_counts = groups.count()
    ss_between = float((group_counts * (group_means - grand_mean) ** 2).sum())
    eta_sq = min(ss_between / ss_total, 1.0)

    p_value = None
    n = len(sub)
    if _HAS_SCIPY and n > groups.ngroups:
        try:
            group_values = [g.values for _, g in groups if len(g) > 0]
            _f_stat, p_value = _scipy_stats.f_oneway(*group_values)
        except Exception:
            p_value = None

    return eta_sq, p_value, groups.ngroups


def _numeric_numeric(sub: pd.DataFrame, col_a: str, col_b: str, config: RelationshipConfig):
    x, y = sub[col_a], sub[col_b]

    candidates = {
        "Pearson": x.corr(y, method="pearson"),
        "Spearman": x.corr(y, method="spearman"),
    }
    if config.use_kendall:
        candidates["Kendall"] = x.corr(y, method="kendall")

    candidates = {k: (0.0 if pd.isna(v) else float(v)) for k, v in candidates.items()}
    method = max(candidates, key=lambda k: abs(candidates[k]))
    value = candidates[method]

    direction = "Positive" if value > 0 else ("Negative" if value < 0 else "None")

    p_value = None
    if _HAS_SCIPY and len(sub) > 2:
        try:
            if method == "Pearson":
                _, p_value = _scipy_stats.pearsonr(x, y)
            elif method == "Spearman":
                _, p_value = _scipy_stats.spearmanr(x, y)
            else:
                _, p_value = _scipy_stats.kendalltau(x, y)
        except Exception:
            p_value = None

    return abs(value), method, direction, p_value


def _datetime_numeric(sub: pd.DataFrame, date_col: str, num_col: str):
    parsed = pd.to_datetime(sub[date_col], errors="coerce")
    ordinal = parsed.map(lambda d: d.toordinal() if pd.notna(d) else np.nan)
    tmp = pd.DataFrame({"d": ordinal, "v": sub[num_col]}).dropna()
    if len(tmp) < 3:
        return None

    corr = tmp["d"].corr(tmp["v"], method="spearman")
    corr = 0.0 if pd.isna(corr) else float(corr)
    if corr > 0.05:
        direction = "Increasing"
    elif corr < -0.05:
        direction = "Decreasing"
    else:
        direction = "Flat"

    p_value = None
    if _HAS_SCIPY and len(tmp) > 2:
        try:
            _, p_value = _scipy_stats.spearmanr(tmp["d"], tmp["v"])
        except Exception:
            p_value = None

    return abs(corr), "Datetime Trend (Spearman)", direction, p_value, len(tmp)


# =====================================================
# MULTIPLE-TESTING CORRECTION (Benjamini-Hochberg)
# =====================================================


def _benjamini_hochberg(p_values: list, alpha: float) -> list:
    """Return a bool list: True where the null hypothesis is rejected."""
    n = len(p_values)
    if n == 0:
        return []

    indexed = [(p, i) for i, p in enumerate(p_values) if p is not None]
    indexed.sort(key=lambda t: t[0])

    reject = [False] * n
    if not indexed:
        return reject

    m = len(indexed)
    max_k = -1
    for rank, (p, _orig_i) in enumerate(indexed, start=1):
        if p <= (rank / m) * alpha:
            max_k = rank

    for rank, (_p, orig_i) in enumerate(indexed, start=1):
        if rank <= max_k:
            reject[orig_i] = True

    return reject


def _classify(strength: float, config: RelationshipConfig) -> tuple:
    if strength >= config.strong_threshold:
        return "Strong", "🔥"
    if strength >= config.moderate_threshold:
        return "Moderate", "🟡"
    if strength >= config.weak_threshold:
        return "Weak", "⚪"
    return "Not Meaningful", "❌"


def _confidence(n: int) -> str:
    if n >= 500:
        return "High"
    if n >= 100:
        return "Medium"
    return "Low"


# =====================================================
# RESULT CONTAINERS
# =====================================================


class RelationshipGraph:
    """
    Nodes + weighted edges for the relationships kept by
    :meth:`RelationshipResult.graph`. Not a rendered image — a plain
    data structure with a readable text/HTML view, so you can either
    read it directly or hand ``.edges`` to networkx/plotly/etc. for
    an actual visual layout.
    """

    def __init__(self, edges: list):
        self.edges = edges  # list of (col_a, col_b, strength, classification)
        self.nodes = sorted({c for e in edges for c in (e[0], e[1])})

    def __len__(self) -> int:
        return len(self.edges)

    def __repr__(self) -> str:
        if not self.edges:
            return "RelationshipGraph(empty — nothing met the threshold)"

        adjacency: dict = {n: [] for n in self.nodes}
        for a, b, strength, cls in self.edges:
            adjacency[a].append((b, strength, cls))
            adjacency[b].append((a, strength, cls))

        lines = [f"RelationshipGraph — {len(self.nodes)} nodes, {len(self.edges)} edges", ""]
        for node in self.nodes:
            neighbors = sorted(adjacency[node], key=lambda t: -t[1])
            if not neighbors:
                continue
            bits = [f"{other} ({strength:.2f})" for other, strength, _cls in neighbors]
            lines.append(f"  {node}")
            lines.append(f"    └── {', '.join(bits)}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        df = pd.DataFrame(
            self.edges, columns=["Column A", "Column B", "Strength", "Classification"]
        )
        return render_html_report(
            {"Relationship Graph (edge list)": df},
            title="Relationship Graph",
            subtitle=f"{len(self.nodes)} nodes, {len(self.edges)} edges",
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.edges, columns=["Column A", "Column B", "Strength", "Classification"]
        )


class RelationshipResult:
    """
    Result of :meth:`RelationshipAnalyzer.analyze`. Holds every
    analyzed pair plus derived views (top-N, matrix, graph, insights,
    dependencies, redundant columns).
    """

    def __init__(
        self,
        results: pd.DataFrame,
        dependencies: pd.DataFrame,
        redundant_groups: list,
        stats: dict,
        config: RelationshipConfig,
    ):
        self._results = results
        self._dependencies = dependencies
        self._redundant_groups = redundant_groups
        self._stats = stats
        self._config = config

    # ---- core views ----

    def summary(self) -> dict:
        """Dict of scan-level counts — columns, pairs, meaningful relationships."""
        return dict(self._stats)

    def top(self, n: int = 10) -> pd.DataFrame:
        """Top ``n`` meaningful relationships, ranked by strength (descending)."""
        meaningful = self._results[self._results["Classification"] != "Not Meaningful"]
        return (
            meaningful.sort_values("Strength", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def matrix(self) -> pd.DataFrame:
        """
        Symmetric matrix of relationship strengths between every
        analyzed column. Not a plain Pearson correlation matrix — the
        value in each cell comes from whichever method was
        appropriate for that pair's semantic types (Pearson/Spearman
        for numeric-numeric, eta-squared for categorical-numeric,
        Cramér's V for categorical-categorical, etc.), so don't
        assume linear-correlation semantics across the whole matrix.
        """
        cols = sorted(set(self._results["Column A"]) | set(self._results["Column B"]))
        arr = np.full((len(cols), len(cols)), np.nan)
        np.fill_diagonal(arr, 1.0)
        mat = pd.DataFrame(arr, index=cols, columns=cols)
        for _, row in self._results.iterrows():
            mat.loc[row["Column A"], row["Column B"]] = row["Strength"]
            mat.loc[row["Column B"], row["Column A"]] = row["Strength"]
        return mat

    def graph(self, top: int = 20, threshold: float | None = None) -> RelationshipGraph:
        """
        A :class:`RelationshipGraph` of the strongest relationships —
        top ``top`` by strength, optionally filtered to only those at
        or above ``threshold``.
        """
        meaningful = self._results[self._results["Classification"] != "Not Meaningful"]
        if threshold is not None:
            meaningful = meaningful[meaningful["Strength"] >= threshold]
        meaningful = meaningful.sort_values("Strength", ascending=False).head(top)

        edges = [
            (r["Column A"], r["Column B"], float(r["Strength"]), r["Classification"])
            for _, r in meaningful.iterrows()
        ]
        return RelationshipGraph(edges)

    def insights(self) -> list:
        """Auto-generated, human-readable observations (see module docstring for scope)."""
        notes = []
        meaningful = self._results[self._results["Classification"] != "Not Meaningful"]

        strong = meaningful[meaningful["Classification"] == "Strong"].sort_values(
            "Strength", ascending=False
        )
        for _, r in strong.head(5).iterrows():
            direction_phrase = {
                "Positive": "tend to increase together",
                "Negative": "tend to move in opposite directions",
                "Increasing": "tends to increase over time",
                "Decreasing": "tends to decrease over time",
            }.get(r["Direction"], "are strongly associated")
            notes.append(
                f"💡 {r['Column A']} and {r['Column B']} {direction_phrase} "
                f"({r['Strength']:.2f}, {r['Method']})."
            )

        negative = meaningful[meaningful["Direction"] == "Negative"].sort_values(
            "Strength", ascending=False
        )
        for _, r in negative.head(3).iterrows():
            notes.append(
                f"⚠ {r['Column A']} and {r['Column B']} are negatively related "
                f"({r['Strength']:.2f}) — worth checking if that's expected."
            )

        if len(self._dependencies):
            for _, r in self._dependencies.head(5).iterrows():
                notes.append(
                    f"🔗 Potential functional dependency: {r['Source']} → {r['Target']} "
                    f"(consistency {r['Consistency %']:.1f}%)."
                )

        for group in self._redundant_groups[:5]:
            notes.append(
                f"♻ Possible redundant/derived columns: {', '.join(group)} — "
                f"they carry very similar information."
            )

        if notes:
            notes.append(
                "⚠ Important: all of the above are statistical associations, "
                "not evidence of causation."
            )
        else:
            notes.append("No strong or clearly meaningful relationships were found.")

        return notes

    def dependencies(self) -> pd.DataFrame:
        """Potential functional dependencies (``Source -> Target``) detected."""
        return self._dependencies

    def redundant_columns(self) -> pd.DataFrame:
        """Groups of columns that appear to carry duplicate/near-duplicate information."""
        return pd.DataFrame(
            {"Columns": [", ".join(g) for g in self._redundant_groups]}
        )

    # ---- export ----

    def to_dataframe(self) -> pd.DataFrame:
        """The full, unfiltered results table — every analyzed pair."""
        return self._results

    def to_json(self) -> str:
        payload = {
            "summary": self._stats,
            "relationships": json.loads(self._results.to_json(orient="records")),
            "dependencies": json.loads(self._dependencies.to_json(orient="records")),
            "redundant_columns": self._redundant_groups,
        }
        return json.dumps(payload, indent=2, default=str)

    def to_html(
        self,
        path: str | None = None,
        title: str = "Relationship Discovery Report",
        style: str = "dashboard",
        top_n: int = 5,
        graph_top: int = 10,
    ) -> str:
        """
        Render this result as a standalone HTML page.

        Parameters
        ----------
        path : str, optional
            If given, writes the HTML to this file path.
        title : str, default "Relationship Discovery Report"
            Page title.
        style : str, default "dashboard"
            ``"dashboard"`` renders a KPI-card + relationship-graph
            dashboard (sidebar-free, standalone). ``"table"`` renders
            the plain multi-section table report shared with the rest
            of drishtipy's HTML reports — useful for very wide result
            sets where a dashboard layout gets crowded.
        top_n : int, default 5
            Rows shown in the "Top Relationships" panel (dashboard
            style only).
        graph_top : int, default 12
            Edges shown in the relationship graph (dashboard style
            only) — kept small deliberately so the graph stays
            readable.

        Returns
        -------
        str
        """
        if style == "dashboard":
            html_text = render_relationship_dashboard(
                self, title=title, top_n=top_n, graph_top=graph_top
            )
        elif style == "table":
            sections = {
                "Top Relationships": self.top(20),
                "All Relationships": self._results,
                "Potential Dependencies": self._dependencies,
                "Redundant Columns": self.redundant_columns(),
                "Insights": pd.DataFrame({"Insight": self.insights()}),
            }
            subtitle = (
                f"{self._stats['columns_analyzed']} columns · "
                f"{self._stats['pairs_analyzed']} pairs analyzed · "
                f"{self._stats['meaningful_relationships']} meaningful relationships"
            )
            html_text = render_html_report(sections, title=title, subtitle=subtitle)
        else:
            raise ValueError(f"Invalid style '{style}'. Must be 'dashboard' or 'table'.")

        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html_text)
        return html_text

    # ---- display ----

    def __repr__(self) -> str:
        s = self._stats
        lines = [
            "",
            "DRISHTIPY — RELATIONSHIP DISCOVERY",
            "━" * 36,
            "",
            f"Columns analyzed:       {s['columns_analyzed']}",
            f"Possible pairs:         {s['possible_pairs']:,}",
            f"Pairs analyzed:         {s['pairs_analyzed']:,}",
            f"Ignored pairs:          {s['ignored_pairs']:,}",
            "",
            f"Meaningful relationships: {s['meaningful_relationships']}",
            f"Strong:                    {s['strong']}",
            f"Moderate:                  {s['moderate']}",
            f"Weak:                      {s['weak']}",
            "",
        ]
        if s.get("is_sampled"):
            lines.append(
                f"(Analyzed {s['analyzed_rows']:,} of {s['total_rows']:,} rows — sampled)"
            )
            lines.append("")

        top5 = self.top(5)
        if len(top5):
            lines.append("TOP RELATIONSHIPS")
            lines.append("")
            for _, r in top5.iterrows():
                _label, emoji = _classify(r["Strength"], self._config)
                lines.append(
                    f"  {r['Column A']:<18} ↔ {r['Column B']:<18} "
                    f"{r['Strength']:.2f} {emoji}"
                )
            lines.append("")

        lines.append("result.top(20) / .matrix() / .graph() / .insights() / .dependencies()")
        lines.append("")
        return "\n".join(lines)


# =====================================================
# ANALYZER
# =====================================================


class RelationshipAnalyzer:
    """
    Discovers, scores, and ranks relationships between every pair of
    columns in a DataFrame that can be meaningfully compared.

    Examples
    --------
    >>> from drishtipy import RelationshipAnalyzer
    >>> analyzer = RelationshipAnalyzer(df)
    >>> result = analyzer.analyze()
    >>> result.top(10)
    >>> result.insights()
    """

    def __init__(self, df: pd.DataFrame, config: RelationshipConfig | None = None):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        self.df = df
        self.config = config or RelationshipConfig()

    def analyze(self) -> RelationshipResult:
        config = self.config
        df = self.df
        total_rows = len(df)
        is_sampled = False

        if config.sample_size is not None and total_rows > config.sample_size:
            df = df.sample(n=config.sample_size, random_state=config.random_state)
            is_sampled = True

        columns = list(df.columns)
        col_types = {c: detect_semantic_type(df[c], c) for c in columns}

        all_pairs = list(itertools.combinations(columns, 2))
        candidate_pairs = []
        ignored_count = 0

        for col_a, col_b in all_pairs:
            type_a, type_b = col_types[col_a], col_types[col_b]

            if type_a == "constant" or type_b == "constant":
                ignored_count += 1
                continue
            if not config.include_id_pairs and (type_a == "id" or type_b == "id"):
                ignored_count += 1
                continue
            if type_a in _EXCLUDED or type_b in _EXCLUDED:
                if not (config.include_id_pairs and type_a == "id" and type_b == "id"):
                    ignored_count += 1
                    continue

            candidate_pairs.append((col_a, col_b, type_a, type_b))

        capped = len(candidate_pairs) > config.max_pairs
        if capped:
            candidate_pairs = candidate_pairs[: config.max_pairs]

        rows = []
        p_values = []
        p_value_row_indices = []

        def _dispatch_type(t: str) -> str:
            # For routing to an analysis method only — ID columns are
            # inherently discrete/unordered, so once a user has opted
            # in via include_id_pairs, treat them like categorical
            # columns for method selection. The *displayed* Type A/B
            # in the results table still shows the real "id" label.
            return "categorical" if t == "id" else t

        for col_a, col_b, type_a, type_b in candidate_pairs:
            disp_a, disp_b = _dispatch_type(type_a), _dispatch_type(type_b)
            sub = df[[col_a, col_b]].dropna()
            n = len(sub)

            base = {
                "Column A": col_a,
                "Column B": col_b,
                "Type A": type_a,
                "Type B": type_b,
                "Sample Size": n,
            }

            if n < config.min_sample_size:
                rows.append(
                    {
                        **base,
                        "Method": None,
                        "Strength": 0.0,
                        "Direction": "N/A",
                        "P-Value": None,
                        "Significance": "Unknown",
                        "Confidence": "Low",
                        "Classification": "Not Meaningful",
                        "Notes": "Not enough valid observations",
                    }
                )
                continue

            strength = method = direction = p_value = None
            notes = ""

            if disp_a in _NUMERIC_LIKE and disp_b in _NUMERIC_LIKE:
                strength, method, direction, p_value = _numeric_numeric(
                    sub, col_a, col_b, config
                )

            elif (disp_a in _CATEGORICAL_LIKE) != (disp_b in _CATEGORICAL_LIKE) and (
                disp_a in _NUMERIC_LIKE or disp_b in _NUMERIC_LIKE
            ):
                cat_col, num_col = (
                    (col_a, col_b) if disp_a in _CATEGORICAL_LIKE else (col_b, col_a)
                )
                result = _eta_squared(sub, cat_col, num_col, config.max_categories)
                if result is None:
                    rows.append(
                        {
                            **base,
                            "Method": "ANOVA (eta²)",
                            "Strength": 0.0,
                            "Direction": "N/A",
                            "P-Value": None,
                            "Significance": "Unknown",
                            "Confidence": "Low",
                            "Classification": "Not Meaningful",
                            "Notes": "Too many categories for a reliable comparison",
                        }
                    )
                    continue
                strength, p_value, _k = result
                method, direction = "ANOVA (eta²)", "N/A"

            elif disp_a in _CATEGORICAL_LIKE and disp_b in _CATEGORICAL_LIKE:
                if sub[col_a].nunique() > config.max_categories or (
                    sub[col_b].nunique() > config.max_categories
                ):
                    rows.append(
                        {
                            **base,
                            "Method": "Cramér's V",
                            "Strength": 0.0,
                            "Direction": "N/A",
                            "P-Value": None,
                            "Significance": "Unknown",
                            "Confidence": "Low",
                            "Classification": "Not Meaningful",
                            "Notes": "Too many categories for a reliable comparison",
                        }
                    )
                    continue
                strength, p_value, table = _cramers_v(sub, col_a, col_b)
                method, direction = "Cramér's V", "N/A"
                mi = _normalized_mutual_info(table)
                notes = f"Normalized Mutual Info: {mi:.2f}"

            elif (disp_a in _DATE_LIKE) != (disp_b in _DATE_LIKE) and (
                disp_a in _NUMERIC_LIKE or disp_b in _NUMERIC_LIKE
            ):
                date_col, num_col = (
                    (col_a, col_b) if disp_a in _DATE_LIKE else (col_b, col_a)
                )
                result = _datetime_numeric(sub, date_col, num_col)
                if result is None:
                    rows.append(
                        {
                            **base,
                            "Method": "Datetime Trend (Spearman)",
                            "Strength": 0.0,
                            "Direction": "N/A",
                            "P-Value": None,
                            "Significance": "Unknown",
                            "Confidence": "Low",
                            "Classification": "Not Meaningful",
                            "Notes": "Not enough valid observations",
                        }
                    )
                    continue
                strength, method, direction, p_value, n = result
                base["Sample Size"] = n

            else:
                ignored_count += 1
                continue

            classification, _emoji = _classify(strength, config)

            row_index = len(rows)
            if p_value is not None:
                p_values.append(p_value)
                p_value_row_indices.append(row_index)

            rows.append(
                {
                    **base,
                    "Method": method,
                    "Strength": round(float(strength), 4),
                    "Direction": direction,
                    "P-Value": round(float(p_value), 6) if p_value is not None else None,
                    "Significance": "Unknown (scipy not installed)"
                    if not _HAS_SCIPY
                    else None,  # filled in after FDR correction below
                    "Confidence": _confidence(n),
                    "Classification": classification,
                    "Notes": notes,
                }
            )

        results_df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=[
                "Column A", "Column B", "Type A", "Type B", "Method", "Strength",
                "Direction", "P-Value", "Significance", "Confidence",
                "Classification", "Sample Size", "Notes",
            ]
        )

        # ---- significance + FDR correction (only where p-values exist) ----
        if _HAS_SCIPY and len(p_values):
            if config.fdr_correction:
                rejected = _benjamini_hochberg(p_values, config.alpha)
            else:
                rejected = [p <= config.alpha for p in p_values]
            for idx, is_significant in zip(p_value_row_indices, rejected):
                results_df.loc[idx, "Significance"] = (
                    "Significant" if is_significant else "Not Significant"
                )
        if _HAS_SCIPY:
            results_df["Significance"] = results_df["Significance"].fillna("Unknown")

        # ---- functional dependencies ----
        dependencies_df = self._scan_dependencies(df, col_types, config)

        # ---- redundant columns ----
        redundant_groups = self._find_redundant_columns(df, results_df, config)

        strong = int((results_df["Classification"] == "Strong").sum())
        moderate = int((results_df["Classification"] == "Moderate").sum())
        weak = int((results_df["Classification"] == "Weak").sum())
        meaningful = strong + moderate + weak

        stats = {
            "columns_analyzed": len(columns),
            "possible_pairs": len(all_pairs),
            "pairs_analyzed": len(candidate_pairs),
            "ignored_pairs": ignored_count,
            "capped_at_max_pairs": capped,
            "meaningful_relationships": meaningful,
            "strong": strong,
            "moderate": moderate,
            "weak": weak,
            "is_sampled": is_sampled,
            "analyzed_rows": len(df),
            "total_rows": total_rows,
            "scipy_available": _HAS_SCIPY,
        }

        return RelationshipResult(
            results_df, dependencies_df, redundant_groups, stats, config
        )

    @staticmethod
    def _scan_dependencies(
        df: pd.DataFrame, col_types: dict, config: RelationshipConfig
    ) -> pd.DataFrame:
        sources = [c for c, t in col_types.items() if t in _DEPENDENCY_SOURCE_TYPES]
        targets = [c for c, t in col_types.items() if t in _DEPENDENCY_TARGET_TYPES]

        rows = []
        for source in sources:
            for target in targets:
                if source == target:
                    continue
                sub = df[[source, target]].dropna()
                if len(sub) < config.min_sample_size:
                    continue

                grouped = sub.groupby(source, observed=True)[target].nunique()
                if len(grouped) < 2:
                    continue

                single_valued_groups = grouped[grouped == 1].index
                consistent_rows = sub[source].isin(single_valued_groups).sum()
                consistency = consistent_rows / len(sub)

                if consistency >= config.functional_dependency_threshold:
                    rows.append(
                        {
                            "Source": source,
                            "Target": target,
                            "Consistency %": round(float(consistency * 100), 2),
                            "Rows Checked": len(sub),
                        }
                    )

        if not rows:
            return pd.DataFrame(
                columns=["Source", "Target", "Consistency %", "Rows Checked"]
            )
        return pd.DataFrame(rows).sort_values("Consistency %", ascending=False).reset_index(
            drop=True
        )

    @staticmethod
    def _find_redundant_columns(
        df: pd.DataFrame, results_df: pd.DataFrame, config: RelationshipConfig
    ) -> list:
        # Exact/near-duplicate columns via a fast content fingerprint.
        fingerprints: dict = {}
        for col in df.columns:
            try:
                fp = int(pd.util.hash_pandas_object(df[col], index=False).sum())
            except TypeError:
                fp = hash(tuple(df[col].astype(str)))
            fingerprints.setdefault(fp, []).append(col)

        groups = [sorted(cols) for cols in fingerprints.values() if len(cols) > 1]

        # High-strength analyzed pairs (numeric or categorical) above the
        # redundancy threshold, merged into the same duplicate-groups via
        # a union-find so transitively-linked columns end up together.
        parent = {c: c for c in df.columns}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for group in groups:
            for c in group[1:]:
                union(group[0], c)

        if len(results_df):
            # Redundancy ("this column is a duplicate/derived copy of
            # that one") only makes sense for same-kind comparisons —
            # two numeric columns or two categorical columns storing
            # near-identical information. A strong categorical<->numeric
            # (ANOVA) or datetime-trend association is a different kind
            # of relationship, not evidence of duplicated data, so it's
            # excluded here even if its strength score is high.
            redundancy_methods = {"Pearson", "Spearman", "Kendall", "Cramér's V"}
            redundant_pairs = results_df[
                (results_df["Strength"] >= config.redundancy_threshold)
                & (results_df["Method"].isin(redundancy_methods))
            ]
            for _, r in redundant_pairs.iterrows():
                union(r["Column A"], r["Column B"])

        clusters: dict = {}
        for c in df.columns:
            clusters.setdefault(find(c), []).append(c)

        return [sorted(v) for v in clusters.values() if len(v) > 1]


# =====================================================
# DASHBOARD-STYLE HTML REPORT
# =====================================================
#
# A dedicated, light-themed dashboard renderer for relationship
# results — KPI cards, a top-relationships list, a relationship
# graph, insight cards, and a dependencies list. Self-contained
# (no JS framework, no external assets); the graph is drawn as
# inline SVG with a circular layout computed from the actual node
# count, rather than hand-placed positions, so it works for any
# number of columns instead of only a fixed demo layout.

_DASHBOARD_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: Inter, -apple-system, "Segoe UI", Arial, sans-serif;
    background: #f5f7fb;
    color: #172033;
    padding: 28px 34px;
}
.topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; flex-wrap: wrap; gap: 12px; }
.title h1 { font-size: 26px; margin-bottom: 5px; }
.title p { color: #64748b; font-size: 14px; }
.badge-meta { background: white; border: 1px solid #e2e8f0; padding: 10px 16px; border-radius: 8px; font-size: 13px; color: #475569; }
.cards { display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 25px; }
.card { background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 18px; flex: 1 1 180px; min-width: 160px; }
.card-label { font-size: 12px; color: #64748b; margin-bottom: 8px; }
.card-value { font-size: 25px; font-weight: 750; }
.card-sub { margin-top: 5px; font-size: 11px; color: #64748b; }
.green { color: #059669; } .orange { color: #d97706; } .blue { color: #2563eb; } .grey { color: #64748b; }
.grid { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }
.grid .panel:first-child { flex: 1 1 380px; }
.grid .panel:last-child { flex: 1.15 1 420px; }
.bottom-grid { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }
.bottom-grid .panel { flex: 1 1 320px; }
.panel { background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; min-width: 0; }
.panel-full { flex: 1 1 100%; }
.panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; gap: 10px; flex-wrap: wrap; }
.panel-header h2 { font-size: 16px; }
.panel-header span { color: #64748b; font-size: 12px; }
.empty-note { color: #94a3b8; font-size: 13px; font-style: italic; padding: 8px 0; }
.relationship { display: flex; align-items: center; gap: 10px; padding: 13px 5px; border-bottom: 1px solid #eef2f7; }
.relationship:last-child { border-bottom: none; }
.relationship .column { flex: 1 1 0; min-width: 0; }
.relationship .arrow { flex: 0 0 20px; }
.relationship .score { flex: 0 0 84px; }
.column { font-size: 13px; font-weight: 600; word-break: break-word; }
.arrow { text-align: center; color: #94a3b8; }
.score { text-align: center; font-weight: 750; padding: 5px 9px; border-radius: 20px; font-size: 12px; white-space: nowrap; }
.strong { background: #dcfce7; color: #166534; }
.moderate { background: #fef3c7; color: #92400e; }
.weak { background: #e2e8f0; color: #475569; }
.graph-wrap { width: 100%; }
.graph-wrap svg { width: 100%; height: auto; display: block; }
.node-box {
    background: white; border: 2px solid #cbd5e1; border-radius: 10px;
    padding: 7px 10px; font-size: 11px; font-weight: 700; text-align: center;
    box-shadow: 0 4px 12px rgba(15,23,42,.07); height: 100%;
    display: flex; align-items: center; justify-content: center;
    font-family: Inter, -apple-system, "Segoe UI", Arial, sans-serif; color: #172033;
    overflow: hidden;
}
.node-box.hub { border-color: #2563eb; background: #eff6ff; }
.insight { padding: 13px; border-radius: 9px; margin-bottom: 10px; font-size: 13px; line-height: 1.5; }
.insight:last-child { margin-bottom: 0; }
.insight strong { display: block; margin-bottom: 3px; }
.insight.blue-box { background: #eff6ff; border-left: 4px solid #3b82f6; }
.insight.yellow-box { background: #fffbeb; border-left: 4px solid #f59e0b; }
.insight.red-box { background: #fef2f2; border-left: 4px solid #ef4444; }
.insight.grey-box { background: #f8fafc; border-left: 4px solid #94a3b8; }
.dependency { display: flex; justify-content: space-between; align-items: center; padding: 13px 5px; border-bottom: 1px solid #eef2f7; font-size: 13px; gap: 10px; }
.dependency:last-child { border-bottom: none; }
.dependency span:first-child { word-break: break-word; }
.dependency span:last-child { color: #059669; font-weight: 700; white-space: nowrap; }
@media (max-width: 1000px) {
    .card { flex-basis: 45%; }
    .grid .panel, .bottom-grid .panel { flex-basis: 100%; }
}
"""


_INSIGHT_PREFIXES = ("💡", "🔗", "⚠", "♻")


def _insight_box_class(insight_text: str) -> tuple:
    if insight_text.startswith("💡"):
        return "blue-box", "Relationship found"
    if insight_text.startswith("🔗"):
        return "yellow-box", "Potential dependency"
    if insight_text.startswith("⚠"):
        return "red-box", "Worth investigating"
    if insight_text.startswith("♻"):
        return "yellow-box", "Possible redundancy"
    return "grey-box", "Note"


def _classification_css_class(classification: str) -> str:
    return {"Strong": "strong", "Moderate": "moderate", "Weak": "weak"}.get(
        classification, "weak"
    )


def _render_relationship_rows(top_df: pd.DataFrame) -> str:
    if top_df.empty:
        return '<p class="empty-note">No meaningful relationships found.</p>'

    rows = []
    for _, r in top_df.iterrows():
        css_class = _classification_css_class(r["Classification"])
        emoji = {"Strong": " 🔥", "Moderate": "", "Weak": ""}.get(r["Classification"], "")
        rows.append(
            f'<div class="relationship">'
            f'<div class="column">{_html.escape(str(r["Column A"]))}</div>'
            f'<div class="arrow">↔</div>'
            f'<div class="column">{_html.escape(str(r["Column B"]))}</div>'
            f'<div class="score {css_class}">{r["Strength"]:.2f}{emoji}</div>'
            f"</div>"
        )
    return "".join(rows)


def _render_insights(insights: list) -> str:
    if not insights:
        return '<p class="empty-note">No insights generated.</p>'
    blocks = []
    for text in insights:
        box_class, label = _insight_box_class(text)
        # Only strip the leading emoji when it's actually one of ours —
        # e.g. the plain "No strong relationships found" fallback has
        # no emoji prefix, and splitting on the first space would
        # otherwise eat its first real word.
        if text.startswith(_INSIGHT_PREFIXES) and " " in text:
            body = text.split(" ", 1)[1]
        else:
            body = text
        blocks.append(
            f'<div class="insight {box_class}"><strong>{_html.escape(label)}</strong>'
            f"{_html.escape(body)}</div>"
        )
    return "".join(blocks)


def _render_dependencies(dependencies_df: pd.DataFrame, limit: int = 8) -> str:
    if dependencies_df.empty:
        return '<p class="empty-note">No potential dependencies detected.</p>'
    rows = []
    for _, r in dependencies_df.head(limit).iterrows():
        rows.append(
            f'<div class="dependency">'
            f'<span>{_html.escape(str(r["Source"]))} → {_html.escape(str(r["Target"]))}</span>'
            f'<span>{r["Consistency %"]:.2f}%</span>'
            f"</div>"
        )
    return "".join(rows)


def _render_redundant(redundant_groups: list) -> str:
    if not redundant_groups:
        return '<p class="empty-note">No redundant/derived column groups detected.</p>'
    rows = []
    for group in redundant_groups:
        rows.append(
            f'<div class="dependency">'
            f'<span>{_html.escape(", ".join(group))}</span>'
            f'<span>{len(group)} columns</span>'
            f"</div>"
        )
    return "".join(rows)


def _render_graph_svg(graph: "RelationshipGraph") -> str:
    nodes = graph.nodes
    edges = graph.edges

    if not nodes:
        return '<p class="empty-note">No meaningful relationships to graph.</p>'

    width, height = 640, 420
    cx, cy = width / 2, height / 2 + 10
    radius = min(width, height) / 2 - 65

    if len(nodes) == 1:
        positions = {nodes[0]: (cx, cy)}
    else:
        positions = {}
        for i, node in enumerate(nodes):
            angle = (2 * math.pi * i) / len(nodes) - math.pi / 2
            positions[node] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

    degree: dict = {n: 0 for n in nodes}
    for a, b, _s, _c in edges:
        degree[a] += 1
        degree[b] += 1
    hub = max(degree, key=degree.get) if degree else None

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
    ]

    # Lines first so nodes render on top of their endpoints.
    for a, b, strength, _cls in edges:
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        svg_parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#94a3b8" stroke-width="{1 + 2 * strength:.2f}" opacity="0.7" />'
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        label = f"{strength:.2f}"
        label_w = 9 * len(label) + 10
        svg_parts.append(
            f'<rect x="{mx - label_w / 2:.1f}" y="{my - 10:.1f}" width="{label_w}" '
            f'height="18" rx="4" fill="#334155" />'
        )
        svg_parts.append(
            f'<text x="{mx:.1f}" y="{my + 3:.1f}" font-size="10" fill="white" '
            f'text-anchor="middle" font-family="Inter, Arial, sans-serif">{_html.escape(label)}</text>'
        )

    for node in nodes:
        x, y = positions[node]
        label = str(node)
        box_w = max(80, min(150, 8 * len(label) + 24))
        box_h = 34
        hub_class = " hub" if node == hub else ""
        svg_parts.append(
            f'<foreignObject x="{x - box_w / 2:.1f}" y="{y - box_h / 2:.1f}" '
            f'width="{box_w}" height="{box_h}">'
            f'<div xmlns="http://www.w3.org/1999/xhtml" class="node-box{hub_class}">'
            f"{_html.escape(label)}</div></foreignObject>"
        )

    svg_parts.append("</svg>")
    return f'<div class="graph-wrap">{"".join(svg_parts)}</div>'


def render_relationship_dashboard(
    result: "RelationshipResult",
    title: str = "Relationship Discovery",
    subtitle: str | None = None,
    top_n: int = 5,
    graph_top: int = 10,
) -> str:
    """Render a RelationshipResult as a standalone, dashboard-style HTML page."""

    stats = result.summary()
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if subtitle is None:
        bits = [f"Generated {generated}"]
        if stats.get("is_sampled"):
            bits.append(
                f"sampled {stats['analyzed_rows']:,} of {stats['total_rows']:,} rows"
            )
        if not stats.get("scipy_available", True):
            bits.append("significance testing unavailable (scipy not installed)")
        subtitle = " · ".join(bits)

    top_rows_html = _render_relationship_rows(result.top(top_n))
    graph_html = _render_graph_svg(result.graph(top=graph_top))
    insights_html = _render_insights(result.insights())
    dependencies_html = _render_dependencies(result.dependencies())
    redundant_groups = result._redundant_groups
    redundant_html = _render_redundant(redundant_groups)

    redundant_panel = ""
    if redundant_groups:
        redundant_panel = f"""
        <section class="panel panel-full">
            <div class="panel-header">
                <div><h2>Redundant / Derived Columns</h2>
                <span>Column groups that appear to store the same information</span></div>
            </div>
            {redundant_html}
        </section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_html.escape(title)}</title>
<style>{_DASHBOARD_CSS}</style>
</head>
<body>

<div class="topbar">
    <div class="title">
        <h1>{_html.escape(title)}</h1>
        <p>Automatically discovered relationships between dataset columns.</p>
    </div>
    <div class="badge-meta">{_html.escape(subtitle)}</div>
</div>

<section class="cards">
    <div class="card">
        <div class="card-label">Columns</div>
        <div class="card-value">{stats['columns_analyzed']}</div>
        <div class="card-sub">Semantic types detected</div>
    </div>
    <div class="card">
        <div class="card-label">Possible Pairs</div>
        <div class="card-value">{stats['possible_pairs']:,}</div>
        <div class="card-sub">n × (n-1) / 2</div>
    </div>
    <div class="card">
        <div class="card-label">Pairs Analyzed</div>
        <div class="card-value blue">{stats['pairs_analyzed']:,}</div>
        <div class="card-sub">{stats['ignored_pairs']:,} irrelevant pairs ignored</div>
    </div>
    <div class="card">
        <div class="card-label">Strong Relations</div>
        <div class="card-value green">{stats['strong']}</div>
        <div class="card-sub">Score ≥ strong threshold</div>
    </div>
    <div class="card">
        <div class="card-label">Moderate</div>
        <div class="card-value orange">{stats['moderate']}</div>
        <div class="card-sub">Weak: {stats['weak']}</div>
    </div>
</section>

<div class="grid">
    <section class="panel">
        <div class="panel-header">
            <div><h2>Top Relationships</h2>
            <span>Most significant column associations</span></div>
        </div>
        {top_rows_html}
    </section>

    <section class="panel">
        <div class="panel-header">
            <div><h2>Relationship Graph</h2>
            <span>Top {graph_top} meaningful connections</span></div>
        </div>
        {graph_html}
    </section>
</div>

<div class="bottom-grid">
    <section class="panel">
        <div class="panel-header">
            <div><h2>Automatic Insights</h2>
            <span>Generated from relationship analysis</span></div>
        </div>
        {insights_html}
    </section>

    <section class="panel">
        <div class="panel-header">
            <div><h2>Potential Dependencies</h2>
            <span>Possible functional relationships</span></div>
        </div>
        {dependencies_html}
    </section>
</div>
{redundant_panel}

</body>
</html>"""
