"""Minimal standalone HTML for GET /deploy-status?format=html.

Hand-rolled (no template engine, no new dependency) — the deploy view is a small
read-only table plus an SVG flowchart, so a plain f-string with HTML-escaped values
is the right amount of tool. Pulling in Jinja2 for this would be over-engineering.

Security note: every value that comes from the run records is passed through
``html.escape`` before being interpolated, so a crafted commit message or path can't
inject markup (defence against stored XSS in an internal dashboard).
"""
from __future__ import annotations

from html import escape

from . import deploy_status

# The palette below maps each abstract node *state* (computed in deploy_status.py) to
# concrete presentation. Keeping colours here — not in the logic module — keeps the
# "what happened" model separate from "how it looks".
# flowchart node fill colour by step state (green=ok, red=failed, amber=warn, blue=in-flight)
_STEP_FILL = {
    "ok": "#2e7d32", "failed": "#c62828", "warn": "#f9a825",
    "running": "#1565c0", "queued": "#1565c0",
    "skipped": "#e0e0e0", "pending": "#bdbdbd",
}
_STEP_TEXT = {"skipped": "#555", "warn": "#3a2f00"}  # dark text on light fills; else white
# short sub-label shown inside a node (full action string lives in the table)
_STEP_LABEL = {
    "ok": "ok", "failed": "failed", "warn": "needs check",
    "running": "running", "queued": "queued", "skipped": "not run", "pending": "—",
}

_STATUS_COLOR = {
    "success": "#2e7d32",
    "tests_failed": "#c62828",
    "ff_failed": "#c62828",
    "fetch_failed": "#c62828",
    "error": "#c62828",
    "unknown": "#757575",
}


def _badge(status: str) -> str:
    """Render an overall-status word as a small coloured pill (e.g. green "success")."""
    color = _STATUS_COLOR.get(status, "#757575")
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:0.85em">{escape(status)}</span>'
    )


def _actions(actions: dict) -> str:
    """Format the per-deploy action map ({restart: ok, trigger: queued, ...}) as a
    compact "key=value · key=value" string for the history table. ``&mdash;`` (an em
    dash) stands in for "no actions recorded"."""
    if not actions:
        return "&mdash;"
    parts = [f"{escape(str(k))}={escape(str(v))}" for k, v in actions.items()]
    return " · ".join(parts)


def _row(rec: dict) -> str:
    """Build one <tr> of the recent-deploys table from a run record.

    Each field is escaped and given a sensible fallback so a sparse record still
    renders cleanly. The commit is shortened to 7 chars (the conventional short SHA).
    """
    commit = escape((rec.get("new_commit") or "")[:7]) or "&mdash;"
    changed = rec.get("changed_paths") or []
    changed_txt = escape(", ".join(changed)) if changed else "&mdash;"
    return (
        "<tr>"
        f"<td>{escape(rec.get('finished_at', '') or '—')}</td>"
        f"<td><code>{commit}</code></td>"
        f"<td>{_badge(rec.get('status', 'unknown'))}</td>"
        f"<td>{escape(str(rec.get('duration_seconds', 0)))}s</td>"
        f"<td style='font-size:0.85em'>{_actions(rec.get('actions', {}))}</td>"
        f"<td style='font-size:0.8em;color:#555'>{changed_txt}</td>"
        "</tr>"
    )


def _svg_node(x: int, y: int, step: dict, w: int = 150, h: int = 46) -> str:
    """Emit one rounded rectangle + two lines of centred text for a flowchart node.

    We draw raw SVG primitives (``<rect>``/``<text>``) rather than pulling in a chart
    library: the diagram is a handful of fixed boxes, so hand-placing them by (x, y)
    is simplest. ``cx`` centres the labels horizontally within the box.
    """
    fill = _STEP_FILL.get(step["state"], "#bdbdbd")
    text = _STEP_TEXT.get(step["state"], "#fff")
    cx = x + w // 2
    name = escape(step["name"])
    detail = escape(_STEP_LABEL.get(step["state"], step["state"]))
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" '
        f'stroke="#00000022"/>'
        f'<text x="{cx}" y="{y + 19}" text-anchor="middle" fill="{text}" '
        f'font-size="13" font-weight="600">{name}</text>'
        f'<text x="{cx}" y="{y + 35}" text-anchor="middle" fill="{text}" '
        f'font-size="10" opacity="0.85">{detail}</text>'
    )


def _arrow(x1: int, y1: int, x2: int, y2: int) -> str:
    """A connector line between two nodes, ending in the arrowhead marker defined once
    in the SVG ``<defs>`` (referenced via ``marker-end``)."""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#9e9e9e" stroke-width="2" marker-end="url(#arrow)"/>'


def render_flowchart_svg(record) -> str:
    """Inline SVG DAG of the deploy pipeline, coloured by the latest run's outcome.

    Layout: three gate steps left-to-right, then the test gate branches into the
    three conditional actions stacked on the right.
    """
    # Ask the logic module for node states, then this function is pure layout: fixed
    # pixel coordinates for the gate chain (left), the action column (right), and the
    # async retrain node (far right). All the "what colour" decisions already happened.
    steps = deploy_status.deploy_steps(record)
    gate, actions = steps["gate"], steps["actions"]
    retrain = steps.get("retrain")
    w, h = 150, 46            # node width/height, reused for arrow anchor maths
    gate_x = [10, 200, 390]   # x of each of the 3 gate nodes, left-to-right
    gate_y = 97               # gate row sits vertically centred in the canvas
    act_x = 620               # the 3 action nodes share this x (a vertical column)
    act_y = [15, 97, 179]     # y of each action node, top-to-bottom
    retrain_x = 810  # sits to the right of the Trigger Airflow action (actions[1])

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 250" width="100%" '
        'style="max-width:960px" role="img" aria-label="deploy pipeline flowchart">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" fill="#9e9e9e"/></marker></defs>',
    ]
    # Draw the 3 gate nodes in a row, joining each to the previous with an arrow (the
    # arrow starts at the right edge of node i-1 and ends at the left edge of node i).
    for i, step in enumerate(gate):
        parts.append(_svg_node(gate_x[i], gate_y, step))
        if i > 0:
            parts.append(_arrow(gate_x[i - 1] + w, gate_y + h // 2, gate_x[i], gate_y + h // 2))
    # The test gate (last gate node) fans out to all three action nodes — one arrow
    # each — visually expressing "these actions run only after the gate passes".
    branch_x = gate_x[2] + w
    branch_y = gate_y + h // 2
    for i, step in enumerate(actions):
        parts.append(_arrow(branch_x, branch_y, act_x, act_y[i] + h // 2))
        parts.append(_svg_node(act_x, act_y[i], step))
    # The retrain node hangs off the middle action ("Trigger Airflow", actions[1])
    # because a retrain is the *async child* of that trigger, resolved later.
    if retrain:
        trig_y = act_y[1] + h // 2
        parts.append(_arrow(act_x + w, trig_y, retrain_x, trig_y))
        parts.append(_svg_node(retrain_x, act_y[1], retrain))
    parts.append("</svg>")
    return "".join(parts)


def render_deploy_html(snapshot: dict) -> str:
    """Assemble the full HTML page from a deploy snapshot: a headline, the pipeline
    flowchart, a colour legend, and the recent-deploys table. Handles the empty case
    (no deploys yet) with an explanatory message instead of a broken layout."""
    latest = snapshot.get("latest")
    recent = snapshot.get("recent", [])
    overall = snapshot.get("status", "unknown")

    if not latest:
        headline = (
            "<p style='color:#757575'>No deploys recorded yet "
            "(the CD hook writes a record on the next commit to <code>main</code>).</p>"
        )
        rows = ""
    else:
        headline = (
            f"<p>Last deploy: {_badge(overall)} &nbsp;"
            f"<code>{escape((latest.get('new_commit') or '')[:7])}</code> &nbsp;"
            f"at {escape(latest.get('finished_at', ''))} &nbsp;"
            f"({escape(str(latest.get('duration_seconds', 0)))}s)</p>"
        )
        rows = "".join(_row(r) for r in recent)

    flowchart = render_flowchart_svg(latest)
    legend = (
        '<div style="font-size:0.75em;color:#666;margin-top:.3rem">'
        '<span style="color:#2e7d32">&#9632;</span> ok &nbsp;'
        '<span style="color:#c62828">&#9632;</span> failed &nbsp;'
        '<span style="color:#f9a825">&#9632;</span> warning &nbsp;'
        '<span style="color:#1565c0">&#9632;</span> queued / running &nbsp;'
        '<span style="color:#bdbdbd">&#9632;</span> skipped / not run</div>'
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>delivery — deploy status</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
 h1 {{ font-size: 1.3rem; }}
 table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
 th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #eee; }}
 th {{ background: #fafafa; font-size: 0.8em; text-transform: uppercase; color: #666; }}
 code {{ background: #f2f2f2; padding: 1px 4px; border-radius: 3px; }}
</style></head>
<body>
<h1>Delivery-risk CD — deploy status</h1>
{headline}
<h2 style="font-size:1rem;margin-top:1.2rem">Last deploy pipeline</h2>
{flowchart}
{legend}
<h2 style="font-size:1rem;margin-top:1.4rem">Recent deploys</h2>
<table>
 <thead><tr><th>Finished</th><th>Commit</th><th>Status</th><th>Duration</th>
 <th>Actions</th><th>Changed paths</th></tr></thead>
 <tbody>{rows}</tbody>
</table>
<p style="margin-top:1rem;font-size:0.8em;color:#888">
 Source: CD-hook run records · JSON at <code>/deploy-status</code></p>
</body></html>"""
