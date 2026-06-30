"""
report.py — HTML report generator for SPS Foundation Settlement.
Entry point: build_report(data, date_idx, metric) → html_str

Data contract (from data_processing.process_all):
  data["columns"]  — list of column dicts (settlements/rates in inches or in/yr)
  data["beams"]    — list of beam dicts  (floor_diffs/grade_beam_diffs in inches)
  data["stats"]    — population statistics, color scale anchors
  Positions in feet.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from collections import defaultdict

# ── Configuration ─────────────────────────────────────────────────────────────
DIFF_CONCERN   = 2.0   # in  beam differential → "Concern"
DIFF_WATCH     = 1.0   # in  beam differential → "Watch"
STRESS_CONCERN = 25.0  # ksi 50 % of Fy — concern threshold
STRESS_WATCH   = 12.5  # ksi 25 % of Fy — watch threshold
EXAG           = 20    # vertical exaggeration (matches app default)
COL_HEIGHT     = 17    # ft  visual pile height (matches Three.js VISUAL_COL_HEIGHT)

_BG = "#0d1117"        # dark background matching Three.js scene


# ── Color helpers (match scene.js) ────────────────────────────────────────────

def _value_to_hex(value, min_val, max_val):
    """Blue → cyan → yellow → red (matches scene.js valueToColor)."""
    if value is None:
        return "#4fc3f7"
    r_v = max_val - min_val or 1.0
    t = min(1.0, max(0.0, (value - min_val) / r_v))
    if t < 0.33:
        r, g, b = 0, int(255 * (0.5 + t * 1.5)), int(255 * (1.0 - t * 3.0))
    elif t < 0.66:
        s = (t - 0.33) / 0.33
        r, g, b = int(255 * s), int(255 * 0.8), int(255 * 0.1)
    else:
        s = (t - 0.66) / 0.34
        r, g, b = 255, int(255 * 0.8 * (1.0 - s)), 0
    return f"#{r:02x}{g:02x}{b:02x}"


def _diff_color(diff_in):
    """Beam differential color (matches scene.js beamDiffToColor)."""
    if diff_in is None:
        return "#607080"
    if diff_in >= 3.0: return "#ff0044"
    if diff_in >= 2.0: return "#ff2200"
    if diff_in >= 1.0: return "#ffcc00"
    return "#00cc00"


def _stress_hex(ksi):
    """Green → yellow → red for bending stress (matches scene.js stressToColor)."""
    if ksi is None:
        return "#607080"
    ratio = max(0.0, ksi) / 50.0
    if ratio <= 0.5:
        t = ratio / 0.5
        r, g, b = int(255 * t), 255, 0
    elif ratio <= 1.0:
        t = (ratio - 0.5) / 0.5
        r, g, b = 255, int(255 * (1 - t)), 0
    else:
        r, g, b = 255, 38, 0
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Statistical helpers ────────────────────────────────────────────────────────

def _sigma(value, mean, std):
    """Return sigma deviation as float, or None if undefined."""
    if std is None or std <= 0 or value is None:
        return None
    return (value - mean) / std


def _sigma_label(sigma):
    if sigma is None: return "—"
    return f"{sigma:+.2f}σ"


def _sigma_row_bg(sigma):
    """Dark-themed row background by sigma level."""
    if sigma is None: return ""
    if sigma > 2:  return "background:#2d1010"
    if sigma > 1:  return "background:#241d00"
    if sigma < -1: return "background:#0d1a0d"
    return ""


def _sigma_badge(sigma):
    if sigma is None: return "—"
    if sigma > 2:
        return '<span style="background:#c0392b;color:white;padding:1px 7px;border-radius:8px;font-size:11px">&gt;+2σ</span>'
    if sigma > 1:
        return '<span style="background:#d68910;color:white;padding:1px 7px;border-radius:8px;font-size:11px">&gt;+1σ</span>'
    if sigma < -1:
        return '<span style="background:#1a7a4a;color:white;padding:1px 7px;border-radius:8px;font-size:11px">&lt;−1σ</span>'
    return '<span style="background:#444;color:white;padding:1px 7px;border-radius:8px;font-size:11px">±1σ</span>'


# ── Plotly dark theme helpers ──────────────────────────────────────────────────

def _dark_2d(fig, title="", height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color="white"), x=0.5),
        plot_bgcolor=_BG, paper_bgcolor=_BG,
        font=dict(color="white", size=11),
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)", x=0.01, y=0.99),
        margin=dict(l=55, r=20, t=35, b=40),
        height=height,
        xaxis=dict(color="white", gridcolor="#2a2a3e", title_font=dict(color="#aaa")),
        yaxis=dict(color="white", gridcolor="#2a2a3e", title_font=dict(color="#aaa")),
    )
    return fig


def _dark_3d(fig, camera, title="", height=520):
    bg = "rgb(13,17,23)"
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color="white")),
        scene=dict(
            camera=camera, bgcolor=bg,
            xaxis=dict(title="", showticklabels=False, showgrid=False,
                       backgroundcolor=bg, showspikes=False),
            yaxis=dict(title="", showticklabels=False, showgrid=False,
                       backgroundcolor=bg, showspikes=False),
            zaxis=dict(title=f"Elev. (×{EXAG} exag.)", showgrid=True,
                       backgroundcolor=bg, color="white",
                       tickfont=dict(color="white", size=9)),
            aspectmode="manual",
            aspectratio=dict(x=3.0, y=1.0, z=0.6),
        ),
        paper_bgcolor=_BG, height=height,
        margin=dict(l=0, r=60, t=40, b=0),
        font=dict(color="white"),
    )
    return fig


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _section(title, content, anchor=""):
    aid = f'id="{anchor}"' if anchor else ""
    return (f'\n<div class="report-section" {aid}>'
            f'\n<h2 class="section-title">{title}</h2>\n{content}\n</div>')


def _subsection(title, content):
    return f'<h3 class="sub-title">{title}</h3>\n{content}'


def _fig_html(fig, caption=""):
    html = pio.to_html(fig, full_html=False, include_plotlyjs=False)
    if caption:
        html += f'<p class="fig-caption">{caption}</p>'
    return html


def _data_table(headers, rows_html):
    thead = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    return (f'<table class="data-table"><thead>{thead}</thead>'
            f'<tbody>{"".join(rows_html)}</tbody></table>')


def _bar_cell(value, max_value, color):
    w = min(60, int(value / max(max_value, 0.001) * 60))
    return (f'<div style="display:flex;align-items:center;gap:5px">'
            f'<div style="width:{w}px;height:8px;background:{color};'
            f'border-radius:2px;flex-shrink:0"></div>'
            f'<span>{value:.2f}</span></div>')


def _diff_badge(diff):
    if diff is None: return "—"
    if diff >= DIFF_CONCERN:
        return '<span class="badge-concern">Concern</span>'
    if diff >= DIFF_WATCH:
        return '<span class="badge-watch">Watch</span>'
    return '<span class="badge-ok">OK</span>'


def _diff_row_bg(diff):
    if diff is None: return ""
    if diff >= DIFF_CONCERN: return "background:#2d1010"
    if diff >= DIFF_WATCH:   return "background:#241d00"
    return ""


def _stress_badge(ksi):
    if ksi is None: return "—"
    if ksi >= STRESS_CONCERN:
        return '<span class="badge-concern">Concern</span>'
    if ksi >= STRESS_WATCH:
        return '<span class="badge-watch">Watch</span>'
    return '<span class="badge-ok">OK</span>'


# ── Plan view (matches app dark heatmap) ─────────────────────────────────────

def _plan_view_fig(data, selected_date, title=""):
    """Top-down heatmap replicating the app's Plan View chart."""
    fig  = go.Figure()
    st   = data["stats"]
    s_min, s_max = st["min_settlement_in"], st["max_settlement_in"] or 1.0

    if selected_date in data["heatmap_grids"]:
        gz = np.array(data["heatmap_grids"][selected_date])
        fig.add_trace(go.Heatmap(
            x=data["heatmap_x"], y=data["heatmap_y"], z=gz,
            colorscale="RdBu_r", zmin=s_min, zmax=s_max,
            colorbar=dict(
                title="Settlement<br>(in)", thickness=10, len=0.8,
                tickfont=dict(size=9, color="white"),
                titlefont=dict(size=9, color="white"),
            ),
            hovertemplate="Settlement: %{z:.2f} in<extra></extra>",
        ))

    for beam in data["beams"]:
        sx, sy = beam.get("start_x"), beam.get("start_y")
        ex, ey = beam.get("end_x"),   beam.get("end_y")
        if None not in (sx, sy, ex, ey):
            fig.add_trace(go.Scatter(
                x=[sx, ex, None], y=[sy, ey, None], mode="lines",
                line=dict(color="rgba(200,200,200,0.2)", width=1),
                hoverinfo="skip", showlegend=False,
            ))

    for col in data["columns"]:
        s = col["settlements"].get(selected_date)
        color = _value_to_hex(s, s_min, s_max) if s is not None else "#888"
        hover = (f"<b>{col['id']}</b><br>Settlement: {s:.2f} in<extra></extra>"
                 if s is not None else f"<b>{col['id']}</b><extra></extra>")
        fig.add_trace(go.Scatter(
            x=[col["x"]], y=[col["y"]], mode="markers+text",
            marker=dict(size=12, color=color,
                        line=dict(width=1.5, color="white"), symbol="square"),
            text=[col["id"]], textposition="top center",
            textfont=dict(size=7, color="white"),
            hovertemplate=hover, showlegend=False,
        ))

    fig.update_layout(
        title=dict(text=title or selected_date, font=dict(size=12, color="white"), x=0.5),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor=_BG, paper_bgcolor=_BG, font=dict(color="white"),
        margin=dict(l=10, r=10, t=35, b=10),
        showlegend=False, height=380,
    )
    return fig


# ── 3D scene render ───────────────────────────────────────────────────────────

def _scene_3d_fig(data, selected_date, camera=None, title="", show_grade_beams=True):
    """
    Dark 3D scene matching the app's Fixed-Datum view.
    Columns positioned by grade-beam elevation; floor beams at column top.
    """
    if camera is None:
        camera = dict(eye=dict(x=1.4, y=-2.2, z=1.8), up=dict(x=0, y=0, z=1))

    fig   = go.Figure()
    st    = data["stats"]
    s_min = st["min_settlement_in"]
    s_max = st["max_settlement_in"] or 1.0
    datum = st.get("datum_grade_beam") or 0.0

    # Grade-beam elevation → Z for each column (Fixed Datum mode)
    col_z_grade = {}   # z at grade level
    col_z_floor = {}   # z at floor/cap level = grade + COL_HEIGHT
    for col in data["columns"]:
        gb = col["grade_beam_elevations"].get(selected_date)
        if gb is not None:
            z_g = (gb - datum) * EXAG
            col_z_grade[col["id"]] = z_g
            col_z_floor[col["id"]] = z_g + COL_HEIGHT

    # Reference grid surface
    fig.add_trace(go.Surface(
        x=[0, 400], y=[0, 130], z=[[0, 0], [0, 0]],
        colorscale=[[0, "rgb(22,30,46)"], [1, "rgb(22,30,46)"]],
        showscale=False, opacity=0.7, hoverinfo="skip",
    ))

    # Batch column sticks (vertical lines from z=0 to grade z) into one trace
    stick_x, stick_y, stick_z = [], [], []
    cap_x, cap_y, cap_z, cap_colors, cap_texts = [], [], [], [], []

    for col in data["columns"]:
        s = col["settlements"].get(selected_date)
        z_g = col_z_grade.get(col["id"])
        z_f = col_z_floor.get(col["id"])
        if z_g is None or s is None:
            continue
        stick_x += [col["x"], col["x"], None]
        stick_y += [col["y"], col["y"], None]
        stick_z += [0, z_f, None]
        cap_x.append(col["x"]); cap_y.append(col["y"]); cap_z.append(z_f)
        cap_colors.append(_value_to_hex(s, s_min, s_max))
        cap_texts.append(f"{col['id']}<br>{s:.2f} in")

    if stick_x:
        fig.add_trace(go.Scatter3d(
            x=stick_x, y=stick_y, z=stick_z, mode="lines",
            line=dict(color="rgba(255,255,255,0.3)", width=2),
            showlegend=False, hoverinfo="skip",
        ))
    if cap_x:
        fig.add_trace(go.Scatter3d(
            x=cap_x, y=cap_y, z=cap_z, mode="markers",
            marker=dict(size=7, color=cap_colors, symbol="square",
                        line=dict(width=1, color="white")),
            text=cap_texts,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

    # Batch floor-level beam lines by color
    floor_pts = defaultdict(lambda: {"x": [], "y": [], "z": []})
    grade_pts = defaultdict(lambda: {"x": [], "y": [], "z": []})

    for beam in data["beams"]:
        if beam.get("is_vlink"): continue
        sx, sy = beam.get("start_x"), beam.get("start_y")
        ex, ey = beam.get("end_x"),   beam.get("end_y")
        if None in (sx, sy, ex, ey): continue

        # Floor beam (connects at column cap height)
        zs_f = col_z_floor.get(beam["start_id"])
        ze_f = col_z_floor.get(beam["end_id"])
        if zs_f is not None and ze_f is not None:
            diff  = beam["floor_diffs"].get(selected_date) or 0
            color = _diff_color(diff)
            floor_pts[color]["x"] += [sx, ex, None]
            floor_pts[color]["y"] += [sy, ey, None]
            floor_pts[color]["z"] += [zs_f, ze_f, None]

        # Grade beam (connects at grade level)
        if show_grade_beams:
            zs_g = col_z_grade.get(beam["start_id"])
            ze_g = col_z_grade.get(beam["end_id"])
            if zs_g is not None and ze_g is not None:
                gbd   = beam["grade_beam_diffs"].get(selected_date) or 0
                color = _diff_color(gbd)
                grade_pts[color]["x"] += [sx, ex, None]
                grade_pts[color]["y"] += [sy, ey, None]
                grade_pts[color]["z"] += [zs_g, ze_g, None]

    for color, pts in floor_pts.items():
        fig.add_trace(go.Scatter3d(
            x=pts["x"], y=pts["y"], z=pts["z"], mode="lines",
            line=dict(color=color, width=3), showlegend=False, hoverinfo="skip",
        ))
    for color, pts in grade_pts.items():
        fig.add_trace(go.Scatter3d(
            x=pts["x"], y=pts["y"], z=pts["z"], mode="lines",
            line=dict(color=color, width=2), showlegend=False, hoverinfo="skip",
        ))

    _dark_3d(fig, camera, title=title or f"Foundation — {selected_date}", height=580)
    return fig


# ── Section 1: Column Settlement Analysis ─────────────────────────────────────

def _beam_max_stress(beam, date):
    sd = beam["grade_beam_stress"].get(date)
    if not isinstance(sd, dict):
        return None
    vals = [v for v in (sd.get("s"), sd.get("e")) if v is not None]
    return max(vals) if vals else None


def _trend_arrow(col):
    """↑ Accel / → Stable / ↓ Decel from multi-window rates."""
    r1  = col.get("rate_1yr",  {})
    r3  = col.get("settlement_rates", {})
    # Use last available values
    v1 = next((v for v in reversed(list(r1.values()))  if v is not None), None)
    v3 = next((v for v in reversed(list(r3.values()))  if v is not None), None)
    if v1 is None or v3 is None or v3 == 0:
        return "—"
    if v1 > v3 * 1.10: return "↑ Accel."
    if v1 < v3 * 0.90: return "↓ Decel."
    return "→ Stable"


def _column_settlement_section(data, selected_date):
    stats = data["stats"]
    ns_stats = stats.get("norm_settle_stats", {}).get(selected_date, {})
    r3_stats = stats.get("rate_3yr_stats",    {}).get(selected_date, {})

    ns_mean = ns_stats.get("mean"); ns_std = ns_stats.get("std")
    r3_mean = r3_stats.get("mean"); r3_std = r3_stats.get("std")

    # ── Normalized settlement table ──────────────────────────────────────────
    ns_rows = []
    for col in data["columns"]:
        install = col.get("installation_date") or "—"
        s  = col["settlements"].get(selected_date)
        ns = col.get("normalized_settlement", {}).get(selected_date)
        r3 = col["settlement_rates"].get(selected_date)
        if ns is None and s is None:
            continue

        age_yr = None
        if install != "—" and selected_date:
            try:
                age_yr = (pd.to_datetime(selected_date) - pd.to_datetime(install)).days / 365.25
            except Exception:
                pass

        sigma_ns = _sigma(ns, ns_mean, ns_std)
        sigma_r3 = _sigma(r3, r3_mean, r3_std)

        ns_rows.append(dict(
            id=col["id"], pod=col["pod"], install=install[:7],
            age=f"{age_yr:.1f}" if age_yr is not None else "—",
            s=s, ns=ns, r3=r3, sigma_ns=sigma_ns, sigma_r3=sigma_r3,
        ))

    ns_rows.sort(key=lambda r: -(r["ns"] or 0))

    trows_ns = []
    for r in ns_rows:
        s_str  = f"{r['s']:.2f}"   if r["s"]  is not None else "—"
        ns_str = f"{r['ns']:.3f}"  if r["ns"] is not None else "—"
        r3_str = f"{r['r3']:.3f}"  if r["r3"] is not None else "—"
        bg     = _sigma_row_bg(r["sigma_ns"])
        trows_ns.append(
            f'<tr style="{bg}">'
            f'<td><b>{r["id"]}</b></td><td>{r["pod"]}</td>'
            f'<td>{r["install"]}</td><td>{r["age"]}</td>'
            f'<td>{s_str}</td>'
            f'<td>{ns_str} {_sigma_badge(r["sigma_ns"])}</td>'
            f'<td>{r3_str} {_sigma_badge(r["sigma_r3"])}</td>'
            f'<td>{_trend_arrow(next((c for c in data["columns"] if c["id"] == r["id"]), {}))}</td>'
            f'</tr>'
        )

    note_mean = (f"Population mean norm. settlement: {ns_mean:.3f} in/yr, "
                 f"σ = {ns_std:.3f} in/yr" if ns_mean is not None else "")

    norm_table_html = (
        f'<p class="note">Normalized settlement = cumulative settlement ÷ column age. '
        f'Removes bias from columns installed at different times. {note_mean}</p>'
        + _data_table(
            ["Column", "Pod", "Install", "Age (yr)", "Settlement (in)",
             "Norm. (in/yr) σ", "3yr Rate (in/yr) σ", "Trend"],
            trows_ns,
        )
    )

    # ── Multi-window rate comparison table ───────────────────────────────────
    rate_rows = []
    for col in data["columns"]:
        r1  = col.get("rate_1yr",  {}).get(selected_date)
        r3  = col["settlement_rates"].get(selected_date)
        r5  = col.get("rate_5yr",  {}).get(selected_date)
        r10 = col.get("rate_10yr", {}).get(selected_date)
        if all(v is None for v in [r1, r3, r5, r10]):
            continue
        accel = None
        if r1 is not None and r3 is not None and r3 != 0:
            accel = r1 - r3
        rate_rows.append(dict(id=col["id"], pod=col["pod"],
                              r1=r1, r3=r3, r5=r5, r10=r10, accel=accel))

    rate_rows.sort(key=lambda r: -(r["r3"] or 0))

    def _rfmt(v): return f"{v:.3f}" if v is not None else "—"
    def _accel_cell(a):
        if a is None: return "—"
        arrow = "↑" if a > 0.01 else ("↓" if a < -0.01 else "→")
        color = "#ff6b6b" if a > 0.01 else ("#6bff9b" if a < -0.01 else "#aaa")
        return f'<span style="color:{color}">{arrow} {a:+.3f}</span>'

    trows_rate = [
        f'<tr><td><b>{r["id"]}</b></td><td>{r["pod"]}</td>'
        f'<td>{_rfmt(r["r1"])}</td><td>{_rfmt(r["r3"])}</td>'
        f'<td>{_rfmt(r["r5"])}</td><td>{_rfmt(r["r10"])}</td>'
        f'<td>{_accel_cell(r["accel"])}</td></tr>'
        for r in rate_rows
    ]

    rate_table_html = (
        '<p class="note">Settlement rate (in/yr) by trailing regression window. '
        'Acceleration = 1yr rate − 3yr rate; positive (↑) means recent rate exceeds medium-term average.</p>'
        + _data_table(
            ["Column", "Pod", "1yr Rate", "3yr Rate", "5yr Rate", "10yr Rate", "Accel. (1yr−3yr)"],
            trows_rate,
        )
    )

    # ── Settlement history chart ─────────────────────────────────────────────
    fig_hist = _settlement_history_fig(data)

    return (
        _subsection("Normalized Settlement (all columns, sorted by lifetime avg. rate)", norm_table_html)
        + _subsection("Multi-Window Rate Comparison", rate_table_html)
        + _subsection("Cumulative Settlement History", _fig_html(fig_hist))
    )


def _settlement_history_fig(data):
    fig        = go.Figure()
    survey_set = set(data["survey_dates"])
    pod_colors = {"A": "#4fc3f7", "B": "#f48fb1"}
    shown_pods = set()

    for col in data["columns"]:
        color  = pod_colors.get(col["pod"], "#aaaaaa")
        obs_d  = [d for d, v in col["settlements"].items() if v is not None and d in survey_set]
        obs_v  = [col["settlements"][d] for d in obs_d]
        proj_d = [d for d, v in col["proj_settlements"].items() if v is not None]
        proj_v = [col["proj_settlements"][d] for d in proj_d]
        if obs_d and proj_d:
            proj_d = [obs_d[-1]] + proj_d
            proj_v = [obs_v[-1]] + proj_v

        show = col["pod"] not in shown_pods
        if show: shown_pods.add(col["pod"])

        fig.add_trace(go.Scatter(
            x=obs_d, y=obs_v, mode="lines",
            name=f"Pod {col['pod']}", legendgroup=col["pod"],
            line=dict(color=color, width=1.5), showlegend=show,
        ))
        if proj_d:
            fig.add_trace(go.Scatter(
                x=proj_d, y=proj_v, mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                legendgroup=col["pod"], showlegend=False,
            ))

    _dark_2d(fig, "Cumulative Settlement — All Monitoring Points", height=380)
    fig.update_layout(yaxis_title="Settlement (in)", xaxis_title="Date")
    return fig


# ── Section 2: Floor Beam Differentials ──────────────────────────────────────

def _beam_diff_rows(beams, selected_date, use_grade_diffs=False):
    rows = []
    for beam in beams:
        diff = (beam["grade_beam_diffs"] if use_grade_diffs else beam["floor_diffs"]).get(selected_date)
        if diff is None: continue
        rows.append(dict(
            id=beam["id"], start=beam["start_id"], end=beam["end_id"],
            diff=diff, length=beam.get("length_ft"),
        ))
    rows.sort(key=lambda r: -r["diff"])
    return rows


def _diff_table_html(rows, max_rows=20, note=""):
    if not rows:
        return '<p class="note">No data at this date.</p>'
    trows = []
    d_max = rows[0]["diff"] if rows else DIFF_CONCERN * 1.5
    for r in rows[:max_rows]:
        trows.append(
            f'<tr style="{_diff_row_bg(r["diff"])}">'
            f'<td><b>{r["id"]}</b></td>'
            f'<td>{r["start"]}</td><td>{r["end"]}</td>'
            f'<td>{_bar_cell(r["diff"], max(d_max, DIFF_CONCERN * 1.5), _diff_color(r["diff"]))}</td>'
            f'<td>{_diff_badge(r["diff"])}</td></tr>'
        )
    out = _data_table(["Beam", "Start", "End", "Differential (in)", "Status"], trows)
    if note:
        out = f'<p class="note">{note}</p>' + out
    return out


def _floor_beam_section(data, selected_date):
    floor_beams = [b for b in data["beams"]
                   if not b.get("is_vlink") and not b.get("is_inter_pod")
                   and not b.get("is_grade")]
    vlinks      = [b for b in data["beams"] if b.get("is_vlink")]
    inter_pods  = [b for b in data["beams"] if b.get("is_inter_pod")]

    note = ("Floor differential between column endpoints. "
            "Green &lt; 1 in | Yellow 1–2 in | Red ≥ 2 in (Concern).")

    floor_rows    = _beam_diff_rows(floor_beams,   selected_date)
    vlink_rows    = _beam_diff_rows(vlinks,         selected_date)
    interpod_rows = _beam_diff_rows(inter_pods,     selected_date)

    html = _subsection(
        "Floor Beams",
        _diff_table_html(floor_rows,    note=note + " Top 20 shown."),
    )
    if interpod_rows:
        html += _subsection(
            "Inter-Pod Connections",
            _diff_table_html(interpod_rows, note="Connections between Pod A and Pod B."),
        )
    html += _subsection(
        "Virtual Links",
        _diff_table_html(vlink_rows,
                         note="Virtual links are topological connectors, not structural beams."),
    )
    return html


# ── Section 3: Grade Beam Analysis ───────────────────────────────────────────

_STRESS_METHOD_HTML = """
<div class="method-box">
<b>Bending Stress Method — Central-Difference Curvature</b><br>
Where three collinear columns share a grade beam, curvature at the middle column is:
<div class="eq">
κ ≈ 2[z<sub>prev</sub>·h<sub>next</sub> + z<sub>next</sub>·h<sub>prev</sub> − z<sub>node</sub>·(h<sub>prev</sub>+h<sub>next</sub>)] / (h<sub>prev</sub>·h<sub>next</sub>·(h<sub>prev</sub>+h<sub>next</sub>))
</div>
Bending stress: <b>σ = E · κ · c</b> &nbsp;(ksi),
where E = 29 000 ksi, c = 14 in (extreme fibre), F<sub>y</sub> = 50 ksi.
Available on floor-survey dates only. End columns without a collinear extension show no value.
</div>
"""


def _grade_beam_section(data, selected_date):
    all_dates   = data["survey_dates"] + data["proj_dates"]
    grade_beams = [b for b in data["beams"] if b.get("is_grade")]

    if not grade_beams:
        return "<p>No grade beam stress data available.</p>"

    # ── Differential table ───────────────────────────────────────────────────
    diff_rows = _beam_diff_rows(grade_beams, selected_date, use_grade_diffs=True)
    diff_html = _diff_table_html(
        diff_rows,
        note="Grade-beam elevation differential between endpoints. Same thresholds as floor beams.",
    )

    # ── Stress over time chart (top 5 beams at selected date) ────────────────
    stress_now = []
    for beam in grade_beams:
        mx = _beam_max_stress(beam, selected_date)
        if mx is not None:
            stress_now.append((mx, beam))
    stress_now.sort(key=lambda t: -t[0])
    top5 = [b for _, b in stress_now[:5]]

    fig_stress = go.Figure()
    for beam in top5:
        xs, ys = [], []
        for d in all_dates:
            v = _beam_max_stress(beam, d)
            if v is not None:
                xs.append(d); ys.append(v)
        if xs:
            fig_stress.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                name=beam["id"], line=dict(width=2),
            ))

    fig_stress.add_hline(y=STRESS_WATCH,   line_dash="dash", line_color="#d68910",
                         annotation_text="Watch (25% Fy)",    annotation_position="right",
                         annotation_font_color="#d68910")
    fig_stress.add_hline(y=STRESS_CONCERN, line_dash="dash", line_color="#c0392b",
                         annotation_text="Concern (50% Fy)",  annotation_position="right",
                         annotation_font_color="#c0392b")
    fig_stress.add_hline(y=50,             line_dash="dot",  line_color="#ff4400",
                         annotation_text="Yield (Fy = 50 ksi)", annotation_position="right",
                         annotation_font_color="#ff4400")

    _dark_2d(fig_stress, "Grade-Beam Bending Stress — Top 5 Beams", height=460)
    fig_stress.update_layout(yaxis_title="Stress (ksi)", xaxis_title="Date")

    # ── Stress ranking table at selected date ────────────────────────────────
    stress_rows = []
    for beam in grade_beams:
        mx = _beam_max_stress(beam, selected_date)
        if mx is None: continue
        stress_rows.append(dict(
            beam=beam["id"], start=beam["start_id"], end=beam["end_id"],
            max_ksi=mx, pct_fy=mx / 50.0 * 100,
        ))
    stress_rows.sort(key=lambda r: -r["max_ksi"])

    trows_s = []
    for r in stress_rows[:15]:
        trows_s.append(
            f'<tr style="{_diff_row_bg(None)}">'
            f'<td><b>{r["beam"]}</b></td>'
            f'<td>{r["start"]}</td><td>{r["end"]}</td>'
            f'<td>{_bar_cell(r["max_ksi"], 50.0, _stress_hex(r["max_ksi"]))}</td>'
            f'<td>{r["pct_fy"]:.0f}%</td>'
            f'<td>{_stress_badge(r["max_ksi"])}</td></tr>'
        )
    stress_table = _data_table(
        ["Beam", "Start", "End", "Max Stress (ksi)", "% Fy", "Status"],
        trows_s,
    )

    return (
        _STRESS_METHOD_HTML
        + _subsection("Grade Beam Differential (at grade level)", diff_html)
        + _subsection("Bending Stress Over Time — Top 5 Beams", _fig_html(fig_stress))
        + _subsection("Bending Stress Ranking at Selected Date", stress_table)
    )


# ── Section 4: Settlement History (cumulative + normalized) ──────────────────

def _history_section(data, selected_date):
    survey_set = set(data["survey_dates"])
    pod_colors = {"A": "#4fc3f7", "B": "#f48fb1"}

    # Normalized settlement over time
    fig_norm = go.Figure()
    shown = set()
    for col in data["columns"]:
        color = pod_colors.get(col["pod"], "#aaa")
        ns    = col.get("normalized_settlement", {})
        dates = [d for d, v in ns.items() if v is not None and d in survey_set]
        vals  = [ns[d] for d in dates]
        if not dates: continue
        show = col["pod"] not in shown
        if show: shown.add(col["pod"])
        fig_norm.add_trace(go.Scatter(
            x=dates, y=vals, mode="lines",
            name=f"Pod {col['pod']}", legendgroup=col["pod"],
            line=dict(color=color, width=1.5), showlegend=show,
        ))

    _dark_2d(fig_norm, "Normalized Settlement (in/yr) — Lifetime Average Rate", height=360)
    fig_norm.update_layout(yaxis_title="Norm. Settlement (in/yr)", xaxis_title="Date")

    fig_hist = _settlement_history_fig(data)
    return (
        _subsection("Cumulative Settlement", _fig_html(fig_hist))
        + _subsection("Normalized Settlement (accounts for different installation dates)",
                      '<p class="note">Columns installed later appear lower here initially; '
                      'once age > 6 months, values reflect in/yr since first measurement.</p>'
                      + _fig_html(fig_norm))
    )


# ── Section 5: Settlement Forecast ───────────────────────────────────────────

def _forecast_section(data, selected_date):
    proj_dates = data["proj_dates"]
    if not proj_dates:
        return "<p>No forecast data available.</p>"

    final_date = proj_dates[-1]
    survey_set = set(data["survey_dates"])
    pod_colors = {"A": "#4fc3f7", "B": "#f48fb1"}
    window     = data.get("forecast_window_start", "—")

    # Plan view at selected date
    plan_fig = _plan_view_fig(data, selected_date,
                              title=f"Settlement Plan — {selected_date}")

    # Forecast chart — top 8 columns by 5yr projected settlement
    ranked = sorted(data["columns"],
                    key=lambda c: -(c["proj_settlements"].get(final_date) or 0))
    top8   = ranked[:8]
    fig_fc = go.Figure()
    shown  = set()
    for col in top8:
        color  = pod_colors.get(col["pod"], "#aaa")
        obs_d  = [d for d, v in col["settlements"].items() if v is not None and d in survey_set]
        obs_v  = [col["settlements"][d] for d in obs_d]
        proj_d = [d for d, v in col["proj_settlements"].items() if v is not None]
        proj_v = [col["proj_settlements"][d] for d in proj_d]
        if obs_d and proj_d:
            proj_d = [obs_d[-1]] + proj_d
            proj_v = [obs_v[-1]] + proj_v
        show = col["id"] not in shown; shown.add(col["id"])
        fig_fc.add_trace(go.Scatter(
            x=obs_d, y=obs_v, mode="lines", name=col["id"],
            line=dict(color=color, width=2), showlegend=show,
        ))
        if proj_d:
            fig_fc.add_trace(go.Scatter(
                x=proj_d, y=proj_v, mode="lines",
                line=dict(color=color, width=2, dash="dot"), showlegend=False,
            ))

    _dark_2d(fig_fc, f"Settlement Forecast — Top 8 Columns", height=380)
    fig_fc.update_layout(yaxis_title="Settlement (in)", xaxis_title="Date")

    # Forecast + 10yr rate comparison table
    # 3yr proj = value at proj_dates[-1] (≈ forecast_years out)
    # 10yr rate = rate_10yr at latest survey date
    latest_survey = data["survey_dates"][-1] if data["survey_dates"] else None
    yr_labels     = [f"~{i+1}yr" for i in range(len(proj_dates))]
    show_idxs     = sorted({0, len(proj_dates) // 2, len(proj_dates) - 1})
    show_dates    = [proj_dates[i] for i in show_idxs]
    show_labels   = [f"~{i+1}yr" for i in show_idxs]

    trows = []
    for col in sorted(data["columns"],
                      key=lambda c: -(c["proj_settlements"].get(final_date) or 0)):
        curr   = col["settlements"].get(latest_survey)
        r3     = col["settlement_rates"].get(latest_survey)
        r10    = col.get("rate_10yr", {}).get(latest_survey)
        proj_v = [col["proj_settlements"].get(d) for d in show_dates]
        vcells = "".join(f"<td>{v:.2f}</td>" if v is not None else "<td>—</td>"
                         for v in proj_v)
        # Trend vs long-term: if 3yr > 10yr → accelerating
        trend_cell = "—"
        if r3 is not None and r10 is not None and r10 > 0:
            ratio = r3 / r10
            if ratio > 1.15:
                trend_cell = f'<span style="color:#ff6b6b">↑ {ratio:.2f}× 10yr rate</span>'
            elif ratio < 0.85:
                trend_cell = f'<span style="color:#6bff9b">↓ {ratio:.2f}× 10yr rate</span>'
            else:
                trend_cell = f'<span style="color:#aaa">≈ {ratio:.2f}× 10yr rate</span>'

        trows.append(
            f"<tr><td><b>{col['id']}</b></td><td>{col['pod']}</td>"
            f"<td>{'—' if curr is None else f'{curr:.2f}'}</td>"
            f"{vcells}"
            f"<td>{'—' if r3 is None else f'{r3:.3f}'}</td>"
            f"<td>{'—' if r10 is None else f'{r10:.3f}'}</td>"
            f"<td>{trend_cell}</td></tr>"
        )

    table = _data_table(
        ["Column", "Pod", "Current (in)"] + [f"Proj. {l}" for l in show_labels]
        + ["3yr Rate", "10yr Rate", "3yr vs 10yr"],
        trows,
    )

    assumptions = (
        f'<p class="note">'
        f'<b>Forecast assumptions:</b> Linear regression from {window} to latest survey '
        f'({data.get("forecast_window_start", "—")}). '
        f'Dotted lines show projected settlement. '
        f'10yr rate is a trailing {10}-year regression slope at the latest survey. '
        f'The "3yr vs 10yr" column shows whether recent settlement rate is faster (↑) '
        f'or slower (↓) than the long-term average.'
        f'</p>'
    )

    return (
        _fig_html(plan_fig)
        + _fig_html(fig_fc)
        + assumptions
        + _subsection("Projection vs Long-Term Rate", table)
    )


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = f"""
body{{font-family:Arial,sans-serif;background:#111827;color:#e2e8f0;font-size:13px}}
h1{{color:#93c5fd;margin-bottom:4px;font-size:20px}}
h2.section-title{{color:#93c5fd;border-bottom:2px solid #1e3a5f;padding-bottom:6px;
  margin-top:36px;margin-bottom:14px;font-size:16px}}
h3.sub-title{{color:#7dd3fc;font-size:13px;margin-top:18px;margin-bottom:8px}}
p.note{{color:#94a3b8;font-size:11px;font-style:italic;margin:4px 0 10px}}
p.fig-caption{{color:#94a3b8;font-size:11px;text-align:center;margin-top:4px}}
.method-box{{background:#0f1929;border:1px solid #1e3a5f;border-radius:4px;
  padding:10px 14px;margin:8px 0 14px;font-size:12px;line-height:1.7}}
.eq{{font-family:monospace;background:#0a1120;padding:6px 12px;border-radius:3px;
  margin:6px 0;font-size:12px;color:#93c5fd}}
.badge-concern{{background:#7f1d1d;color:#fca5a5;padding:1px 7px;
  border-radius:8px;font-size:11px;font-weight:bold}}
.badge-watch{{background:#78350f;color:#fcd34d;padding:1px 7px;
  border-radius:8px;font-size:11px;font-weight:bold}}
.badge-ok{{background:#14532d;color:#86efac;padding:1px 7px;
  border-radius:8px;font-size:11px;font-weight:bold}}
.data-table{{width:100%;border-collapse:collapse;font-size:12px;margin:8px 0}}
.data-table th{{background:#1e3a5f;color:#93c5fd;padding:6px 8px;text-align:left}}
.data-table td{{padding:5px 8px;border-bottom:1px solid #1e293b;vertical-align:middle;
  color:#e2e8f0}}
.data-table tr:hover{{filter:brightness(1.08)}}
.report-section{{margin-bottom:32px}}
@media print{{
  .report-section{{page-break-inside:avoid}}
  h2.section-title{{page-break-before:always}}
}}
"""


# ── Entry point ───────────────────────────────────────────────────────────────

def build_report(data, date_idx, metric):
    all_dates     = data["survey_dates"] + data["proj_dates"]
    selected_date = all_dates[date_idx]
    timestamp     = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    window        = data.get("forecast_window_start", "—")

    # Opening 3D scene — isometric, NE oblique (matches app default camera)
    scene_fig = _scene_3d_fig(
        data, selected_date,
        camera=dict(eye=dict(x=1.4, y=-2.2, z=1.8), up=dict(x=0, y=0, z=1)),
        title=f"Foundation — {selected_date} (Fixed Datum, ×{EXAG} vertical exaggeration)",
    )

    body = "".join([
        _section("1. Column Settlement Analysis",
                 _column_settlement_section(data, selected_date),
                 "columns"),
        _section("2. Floor Beam Differentials",
                 _floor_beam_section(data, selected_date),
                 "floor"),
        _section("3. Grade Beam Analysis",
                 _grade_beam_section(data, selected_date),
                 "grade"),
        _section("4. Settlement History",
                 _history_section(data, selected_date),
                 "history"),
        _section("5. Settlement Forecast",
                 _forecast_section(data, selected_date),
                 "forecast"),
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SPS Foundation Settlement Report — {selected_date}</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>{_CSS}</style>
</head>
<body style="max-width:1100px;margin:0 auto;padding:24px 32px">

<div style="border-bottom:2px solid #1e3a5f;padding-bottom:12px;margin-bottom:20px">
  <h1>Amundsen-Scott South Pole Station — Foundation Settlement Report</h1>
  <p style="color:#94a3b8;margin:2px 0;font-size:12px">
    <b>Survey date:</b> {selected_date} &nbsp;|&nbsp;
    <b>Generated:</b> {timestamp} &nbsp;|&nbsp;
    <b>Forecast regression from:</b> {window}
  </p>
  <p style="color:#64748b;font-size:11px;margin:4px 0 0">
    Beam thresholds — Concern ≥ {DIFF_CONCERN} in differential, Watch ≥ {DIFF_WATCH} in &nbsp;|&nbsp;
    Stress — Concern ≥ {STRESS_CONCERN} ksi ({int(STRESS_CONCERN/50*100)}% Fy),
    Watch ≥ {STRESS_WATCH} ksi ({int(STRESS_WATCH/50*100)}% Fy)
  </p>
</div>

<div style="margin-bottom:24px">
{_fig_html(scene_fig, "Full 3D scene — beam lines colored by floor differential, columns colored by cumulative settlement.")}
</div>

{body}

</body>
</html>"""
