import base64
import io
import json
import os

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ClientsideFunction, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from data_processing import load_beam_info, process_all, read_excel_sheet
from report import build_report

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
EXTERNAL_SCRIPTS = [
    "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js",
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js",
]

app = dash.Dash(
    __name__,
    external_scripts=EXTERNAL_SCRIPTS,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="SPS Foundation Settlement",
)
server = app.server

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
CONTROLS_CARD = dbc.Card([
    dbc.CardBody([
        # Date slider — most-used control, full width at top
        dbc.Row([
            dbc.Col([
                html.Label("Survey / Forecast Date", className="text-muted small"),
                dcc.Slider(
                    id="date-slider",
                    min=0, max=1, step=1, value=0,
                    marks={},
                    disabled=True,
                    tooltip={"placement": "bottom", "always_visible": False},
                    updatemode="drag",
                ),
            ], width=12),
        ], className="mb-3"),

        # Reference planes row
        dbc.Row([
            dbc.Col([
                html.Span("Reference Planes", className="text-muted small"),
                dbc.Checklist(
                    id="plane-toggles",
                    options=[
                        {"label": "Mean",       "value": "mean"},
                        {"label": "Fit — All",  "value": "all"},
                        {"label": "Fit — Pod A","value": "podA"},
                        {"label": "Fit — Pod B","value": "podB"},
                    ],
                    value=[],
                    inline=True,
                    className="small mt-1",
                ),
            ], width=12),
        ], className="mb-2"),

        # Secondary controls — 4 equal columns
        dbc.Row([
            # 3D view mode
            dbc.Col([
                html.Div([
                    html.Span("3D View Mode", className="text-muted small"),
                    html.Span(" ⓘ", id="view-mode-tip",
                              style={"cursor": "help", "color": "#777", "fontSize": "11px"}),
                ]),
                dbc.Tooltip([
                    html.B("Settlement Bowl"), ": piles positioned by cumulative movement from "
                    "their first reading. Best for visualising how the settlement pattern evolves.",
                    html.Br(), html.Br(),
                    html.B("Fixed Datum"), ": piles positioned at their actual grade beam elevation "
                    "relative to a constant reference (mean elevation at the first floor survey). "
                    "Columns that settle drop over time; none can appear to rise.",
                    html.Br(), html.Br(),
                    html.B("Relative to Mean"), ": same as Fixed Datum but re-centred on the "
                    "current mean each frame. Columns settling slower than average appear to rise "
                    "— useful for differential comparison.",
                ], target="view-mode-tip", placement="bottom"),
                dbc.RadioItems(
                    id="view-mode",
                    options=[
                        {"label": "Settlement Bowl",  "value": "bowl"},
                        {"label": "Fixed Datum",      "value": "fixed"},
                        {"label": "Relative to Mean", "value": "elevation"},
                    ],
                    value="fixed",
                    inline=False,
                    className="small mt-1",
                ),
                html.Div([
                    html.Label("Datum elev. (ft)",
                               className="text-muted",
                               style={"fontSize": "10px", "marginBottom": "2px"}),
                    dcc.Input(
                        id="datum-elevation",
                        type="number",
                        value=None,
                        debounce=True,
                        placeholder="auto",
                        className="bg-dark text-light border-secondary form-control form-control-sm",
                        style={"width": "110px"},
                    ),
                ], className="mt-1"),
            ], width=3),

            # Vertical exaggeration
            dbc.Col([
                html.Label("Vertical Exaggeration", className="text-muted small"),
                dcc.Slider(
                    id="exaggeration-slider",
                    min=1, max=20, step=1, value=20,
                    marks={1: "1×", 5: "5×", 10: "10×", 15: "15×", 20: "20×"},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], width=3),

            # Pile color metric
            dbc.Col([
                html.Div([
                    html.Span("Pile Color", className="text-muted small"),
                    html.Span(" ⓘ", id="pile-color-tip",
                              style={"cursor": "help", "color": "#777", "fontSize": "11px"}),
                ]),
                dbc.Tooltip(
                    "Color each pile by the selected metric. "
                    "Settlement rate uses a 3-year trailing linear regression. "
                    "Beam differential coloring: green < 1 in, yellow 1–2 in, red ≥ 2 in.",
                    target="pile-color-tip", placement="bottom",
                ),
                dbc.RadioItems(
                    id="metric-dropdown",
                    options=[
                        {"label": "None",                       "value": "none"},
                        {"label": "Cumulative Settlement",      "value": "settlement"},
                        {"label": "Settlement Rate (3-yr reg)", "value": "rate"},
                    ],
                    value="settlement",
                    inline=False,
                    className="small mt-1",
                ),
                html.Label("Beam Color", className="text-muted small mt-2"),
                dbc.RadioItems(
                    id="beam-color-mode",
                    options=[
                        {"label": "Differential", "value": "differential"},
                        {"label": "Elevation",    "value": "elevation"},
                        {"label": "Stress",       "value": "stress"},
                        {"label": "Gray",         "value": "gray"},
                    ],
                    value="differential",
                    inline=False,
                    className="small mt-1",
                ),
                html.Label("Beam Layers", className="text-muted small mt-2"),
                dbc.Checklist(
                    id="beam-layer-toggles",
                    options=[
                        {"label": "Floor",  "value": "floor"},
                        {"label": "Virtual Link",  "value": "virtual link"},
                        {"label": "Grade",  "value": "grade"},
                    ],
                    value=["floor", "virtual link", "grade"],
                    inline=True,
                    className="small mt-1",
                ),
            ], width=3),

            # Color scale range (user-editable, auto-set from data)
            dbc.Col([
                html.Label("Color Scale Range", className="text-muted small"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Min", className="text-muted",
                                   style={"fontSize": "10px", "marginBottom": "2px"}),
                        dcc.Input(
                            id="color-range-min", type="number", value=0,
                            debounce=True,
                            className="bg-dark text-light border-secondary form-control form-control-sm",
                        ),
                    ], width=6),
                    dbc.Col([
                        html.Label("Max", className="text-muted",
                                   style={"fontSize": "10px", "marginBottom": "2px"}),
                        dcc.Input(
                            id="color-range-max", type="number", value=80,
                            debounce=True,
                            className="bg-dark text-light border-secondary form-control form-control-sm",
                        ),
                    ], width=6),
                ], className="mt-1"),
            ], width=3),
        ]),
    ])
], className="mb-2", style={"background": "#1e2130", "border": "1px solid #2d3250"})

SPARKLINE_CARD = dbc.Card([
    dbc.CardBody([
        html.H6("Settlement History — click a column in the 3D view to select", id="sparkline-title",
                className="text-muted"),
        dcc.Graph(id="sparkline-chart", style={"height": "220px"}, config={"displayModeBar": False}),
    ])
], style={"background": "#1e2130", "border": "1px solid #2d3250"})

app.layout = dbc.Container(fluid=True, children=[
    # ---- header ----
    dbc.Row([
        dbc.Col([
            html.H3("Amundsen-Scott South Pole Station", className="mb-0 mt-3"),
            html.P("Foundation Settlement Analysis", className="text-muted"),
        ], width=8),
        dbc.Col([
            dbc.Row([
                dbc.Col(dcc.Upload(
                    id="upload-xl",
                    children=dbc.Button("Upload Survey Excel", color="primary", size="sm"),
                    multiple=False,
                ), width="auto"),
                dbc.Col(dbc.Button("Compute", id="btn-compute", color="success", size="sm",
                                   disabled=True), width="auto"),
                dbc.Col(dbc.Button("Export Report", id="btn-export", color="secondary", size="sm",
                                   disabled=True), width="auto"),
                dbc.Col(dcc.Download(id="download-report"), width="auto"),
            ], align="center", className="mt-3 justify-content-end"),
        ], width=4),
    ], className="mb-2"),

    # ---- forecast options ----
    dbc.Row([
        dbc.Col([
            html.Label("Years used for forecast:", className="text-muted small"),
            dcc.Input(id="input-forecast-years", type="number", value=3, min=1, max=10,
                      style={"width": "70px", "marginLeft": "8px"},
                      className="bg-dark text-light border-secondary"),
        ], width=3),
        dbc.Col([
            html.Label("Years to forecast:", className="text-muted small"),
            dcc.Input(id="input-nyears", type="number", value=5, min=1, max=20,
                      style={"width": "70px", "marginLeft": "8px"},
                      className="bg-dark text-light border-secondary"),
        ], width=3),
        dbc.Col([
            html.Div(id="status-msg", className="text-muted small mt-1"),
        ], width=6),
    ], className="mb-2"),

    # ---- data stores & polling interval ----
    dcc.Store(id="scene-data"),
    dcc.Store(id="selected-mp", data=None),
    dcc.Store(id="three-dummy"),
    dcc.Interval(id="click-poll", interval=400, n_intervals=0),

    # ---- 3D Foundation View (full width) ----
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    dbc.Row([
                        dbc.Col("3D Foundation View", className="small"),
                        dbc.Col(
                            html.Small("Drag to orbit · Scroll to zoom · Right-drag to pan",
                                       className="text-muted"),
                            width="auto"),
                    ]),
                    className="py-1",
                ),
                dbc.CardBody([
                    html.Div(
                        id="three-canvas-container",
                        style={"width": "100%", "height": "600px", "position": "relative",
                               "background": "#0d1117"},
                        children=[
                            html.Div(
                                id="three-date-label",
                                style={
                                    "position": "absolute",
                                    "top": "8px",
                                    "left": "0",
                                    "right": "0",
                                    "textAlign": "center",
                                    "color": "rgba(255,255,255,0.9)",
                                    "fontSize": "14px",
                                    "fontWeight": "bold",
                                    "pointerEvents": "none",
                                    "zIndex": "50",
                                    "textShadow": "0 1px 4px rgba(0,0,0,0.9)",
                                    "letterSpacing": "0.04em",
                                },
                            ),
                            html.Div(
                                id="three-tooltip",
                                style={
                                    "position": "absolute",
                                    "display": "none",
                                    "background": "rgba(15,20,35,0.95)",
                                    "color": "white",
                                    "padding": "6px 10px",
                                    "borderRadius": "4px",
                                    "fontSize": "12px",
                                    "pointerEvents": "none",
                                    "zIndex": "100",
                                    "border": "1px solid #2d3a55",
                                    "lineHeight": "1.7",
                                    "whiteSpace": "nowrap",
                                },
                            ),
                        ],
                    ),
                ], className="p-0"),
            ], style={"background": "#1e2130", "border": "1px solid #2d3250"}),
        ], width=12),
    ], className="mb-2"),

    # ---- controls ----
    CONTROLS_CARD,

    # ---- heatmap + sparkline ----
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Plan View — Settlement Heatmap", className="py-1 small"),
                dbc.CardBody([
                    dcc.Graph(id="heatmap-chart", style={"height": "480px"},
                              config={"displayModeBar": True, "scrollZoom": True}),
                ], className="p-1"),
            ], style={"background": "#1e2130", "border": "1px solid #2d3250"}),
        ], width=6),
        dbc.Col([
            SPARKLINE_CARD,
        ], width=6),
    ], className="mb-2"),

    # ---- normalized settlement analysis ----
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Column Analysis — Normalized Settlement & Rate Comparison",
                               className="py-1 small"),
                dbc.CardBody([
                    html.Div(id="analysis-table",
                             style={"maxHeight": "360px", "overflowY": "auto",
                                    "fontSize": "12px"}),
                ], className="p-2"),
            ], style={"background": "#1e2130", "border": "1px solid #2d3250"}),
        ], width=7),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Settlement Rate — Multi-Window Comparison",
                               className="py-1 small"),
                dbc.CardBody([
                    dcc.Graph(id="rate-compare-chart", style={"height": "340px"},
                              config={"displayModeBar": False}),
                ], className="p-1"),
            ], style={"background": "#1e2130", "border": "1px solid #2d3250"}),
        ], width=5),
    ], className="mb-2"),

    # ---- hidden div needed for clientside output ----
    html.Div(id="three-canvas-trigger", style={"display": "none"}),
], style={"background": "#0d1117", "minHeight": "100vh"})


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _settlement_color(value, max_val):
    """Blue→yellow→red hex color for heatmap column markers."""
    if value is None or max_val == 0:
        return "#4fc3f7"
    t = min(1.0, value / max(max_val, 0.001))
    if t < 0.5:
        r = int(255 * (t * 2))
        g = int(255 * (0.76 + t * 0.24))
        b = int(255 * (1 - t * 2))
    else:
        r, g, b = 255, int(255 * (1 - (t - 0.5) * 2)), 0
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Python callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("btn-compute", "disabled"),
    Input("upload-xl", "contents"),
)
def enable_compute(contents):
    return contents is None


@app.callback(
    Output("scene-data", "data"),
    Output("status-msg", "children"),
    Output("btn-export", "disabled"),
    Input("btn-compute", "n_clicks"),
    State("upload-xl", "contents"),
    State("upload-xl", "filename"),
    State("input-forecast-years", "value"),
    State("input-nyears", "value"),
    prevent_initial_call=True,
)
def compute_settlement(n_clicks, contents, filename, forecast_years, nyears):
    if not contents:
        raise PreventUpdate

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    xl_bytes = io.BytesIO(decoded)

    try:
        survey = read_excel_sheet(xl_bytes, "SURVEY DATA")
        xl_bytes.seek(0)
        truss = read_excel_sheet(xl_bytes, "TRUSS DATA")
        xl_bytes.seek(0)
        shim = read_excel_sheet(xl_bytes, "SHIM DATA")

        mp_locations, beams = load_beam_info()
        data = process_all(survey, truss, shim, mp_locations, beams,
                           forecast_years=int(forecast_years), nyears=int(nyears))

        n_surveys = len(data["survey_dates"])
        n_floor = len(data["floor_dates"])
        msg = f"Loaded {filename} — {n_surveys} survey dates, {n_floor} floor elevation dates"
        return data, msg, False

    except Exception as e:
        return None, f"Error: {e}", True


@app.callback(
    Output("date-slider", "min"),
    Output("date-slider", "max"),
    Output("date-slider", "value"),
    Output("date-slider", "marks"),
    Output("date-slider", "disabled"),
    Input("scene-data", "data"),
    prevent_initial_call=True,
)
def build_date_slider(data):
    if not data:
        raise PreventUpdate
    dates = data["survey_dates"] + data["proj_dates"]
    n_survey = len(data["survey_dates"])
    marks = {i: {"label": d[:7], "style": {"fontSize": "10px",
                  "color": "#aaa" if i < n_survey else "#f0ad4e"}}
             for i, d in enumerate(dates)
             if i % max(1, len(dates) // 10) == 0 or i == len(dates) - 1}
    return 0, len(dates) - 1, n_survey - 1, marks, False


@app.callback(
    Output("datum-elevation", "value"),
    Input("scene-data", "data"),
    prevent_initial_call=True,
)
def update_datum(data):
    if not data:
        raise PreventUpdate
    return data["stats"].get("datum_grade_beam")


@app.callback(
    Output("color-range-min", "value"),
    Output("color-range-max", "value"),
    Input("scene-data", "data"),
    Input("metric-dropdown", "value"),
    prevent_initial_call=True,
)
def update_color_range(data, metric):
    if not data:
        raise PreventUpdate
    if metric == "settlement":
        return 0, max(1, round(data["stats"]["max_settlement_in"] + 1))
    if metric == "rate":
        mean = data["stats"]["rate_mean"]
        std  = data["stats"]["rate_std"]
        return round(mean - std, 3), round(mean + std, 3)
    return 0, 80


@app.callback(
    Output("heatmap-chart", "figure"),
    Input("scene-data", "data"),
    Input("date-slider", "value"),
    prevent_initial_call=True,
)
def update_heatmap(data, date_idx):
    if not data:
        raise PreventUpdate

    all_dates = data["survey_dates"] + data["proj_dates"]
    selected_date = all_dates[date_idx]
    is_proj = selected_date in data["proj_dates"]

    fig = go.Figure()

    # Heatmap layer (settlement only — grid data not available for projected dates)
    if selected_date in data["heatmap_grids"]:
        gz = np.array(data["heatmap_grids"][selected_date])
        fig.add_trace(go.Heatmap(
            x=data["heatmap_x"],
            y=data["heatmap_y"],
            z=gz,
            colorscale="RdBu_r",
            zmin=data["stats"]["min_settlement_in"],
            zmax=data["stats"]["max_settlement_in"],
            colorbar=dict(title="Settlement (in)", thickness=12, len=0.8,
                          tickfont=dict(size=10), titlefont=dict(size=10)),
            hovertemplate="Settlement: %{z:.2f} in<extra></extra>",
        ))

    # Beam lines
    for beam in data["beams"]:
        sx, sy = beam.get("start_x"), beam.get("start_y")
        ex, ey = beam.get("end_x"), beam.get("end_y")
        if None not in (sx, sy, ex, ey):
            fig.add_trace(go.Scatter(
                x=[sx, ex, None], y=[sy, ey, None],
                mode="lines",
                line=dict(color="rgba(200,200,200,0.3)", width=1),
                hoverinfo="skip",
                showlegend=False,
            ))

    # Monitoring point markers
    for col in data["columns"]:
        val = col["settlements"].get(selected_date)
        if val is None and is_proj:
            val = col["proj_settlements"].get(selected_date)
        color = _settlement_color(val, data["stats"]["max_settlement_in"]) if val is not None else "#888"
        fig.add_trace(go.Scatter(
            x=[col["x"]], y=[col["y"]],
            mode="markers+text",
            marker=dict(size=14, color=color, line=dict(width=1.5, color="white"),
                        symbol="square"),
            text=[col["id"]],
            textposition="top center",
            textfont=dict(size=8, color="white"),
            customdata=[col["id"]],
            hovertemplate=f"<b>{col['id']}</b><br>Settlement: {val:.2f} in<extra></extra>"
                          if val is not None else f"<b>{col['id']}</b><extra></extra>",
            showlegend=False,
        ))

    proj_note = " (projected)" if is_proj else ""
    fig.update_layout(
        title=dict(text=f"{selected_date}{proj_note}", font=dict(size=12), x=0.5),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#1e2130",
        font=dict(color="white"),
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
    )
    return fig


@app.callback(
    Output("sparkline-chart", "figure"),
    Output("sparkline-title", "children"),
    Input("selected-mp", "data"),
    State("scene-data", "data"),
    prevent_initial_call=True,
)
def update_sparkline(mp_id, data):
    if not data or not mp_id:
        raise PreventUpdate

    col = next((c for c in data["columns"] if c["id"] == mp_id), None)
    if col is None:
        raise PreventUpdate

    obs_dates = [d for d, v in col["settlements"].items() if v is not None]
    obs_vals = [col["settlements"][d] for d in obs_dates]

    # forecast_line spans regression window start → last projected date,
    # overlapping the observed period so you can see the fit vs. actual data.
    fl_items = [(d, v) for d, v in col.get("forecast_line", {}).items() if v is not None]
    fl_dates = [it[0] for it in fl_items]
    fl_vals  = [it[1] for it in fl_items]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=obs_dates, y=obs_vals,
        mode="lines+markers",
        name="Observed",
        line=dict(color="#4fc3f7", width=2),
        marker=dict(size=5),
    ))
    if fl_dates:
        fig.add_trace(go.Scatter(
            x=fl_dates, y=fl_vals,
            mode="lines",
            name="Regression / Forecast",
            line=dict(color="#f0ad4e", width=2, dash="dash"),
        ))

    fig.update_layout(
        xaxis=dict(title="", color="white", gridcolor="#2a2a3e"),
        yaxis=dict(title="Settlement (in)", color="white", gridcolor="#2a2a3e"),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#1e2130",
        font=dict(color="white", size=11),
        margin=dict(l=50, r=20, t=10, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1, font=dict(size=10)),
    )
    return fig, f"Settlement History — {mp_id} (Pod {mp_id[0]})"


@app.callback(
    Output("download-report", "data"),
    Input("btn-export", "n_clicks"),
    State("scene-data", "data"),
    State("date-slider", "value"),
    State("metric-dropdown", "value"),
    prevent_initial_call=True,
)
def export_report(n_clicks, data, date_idx, metric):
    if not data:
        raise PreventUpdate
    html_str = build_report(data, date_idx, metric)
    return dict(content=html_str, filename="SPS_Settlement_Report.html", type="text/html")


@app.callback(
    Output("selected-mp", "data", allow_duplicate=True),
    Input("heatmap-chart", "clickData"),
    prevent_initial_call=True,
)
def heatmap_select_mp(click_data):
    if not click_data or not click_data.get("points"):
        raise PreventUpdate
    mp_id = click_data["points"][0].get("customdata")
    if not mp_id:
        raise PreventUpdate
    return mp_id


# ---------------------------------------------------------------------------
# Analysis section callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("analysis-table", "children"),
    Input("scene-data", "data"),
    Input("date-slider", "value"),
    prevent_initial_call=True,
)
def update_analysis_table(data, date_idx):
    if not data:
        raise PreventUpdate
    all_dates     = data["survey_dates"] + data["proj_dates"]
    selected_date = all_dates[date_idx]
    stats         = data["stats"]
    ns_stats      = stats.get("norm_settle_stats", {}).get(selected_date, {})
    r3_stats      = stats.get("rate_3yr_stats",    {}).get(selected_date, {})
    ns_mean       = ns_stats.get("mean")
    ns_std        = ns_stats.get("std")
    r3_mean       = r3_stats.get("mean")
    r3_std        = r3_stats.get("std")

    rows = []
    for col in data["columns"]:
        ns  = (col.get("normalized_settlement") or {}).get(selected_date)
        r3  = col["settlement_rates"].get(selected_date)
        r1  = (col.get("rate_1yr")  or {}).get(selected_date)
        r10 = (col.get("rate_10yr") or {}).get(selected_date)
        if ns is None and r3 is None:
            continue
        sig_ns = (ns - ns_mean) / ns_std if (ns is not None and ns_std and ns_std > 0) else None
        sig_r3 = (r3 - r3_mean) / r3_std if (r3 is not None and r3_std and r3_std > 0) else None
        rows.append(dict(id=col["id"], pod=col["pod"],
                         ns=ns, r1=r1, r3=r3, r10=r10,
                         sig_ns=sig_ns, sig_r3=sig_r3))

    rows.sort(key=lambda r: -(r["ns"] or 0))

    def _fmt(v, d=3): return f"{v:.{d}f}" if v is not None else "—"
    def _sig_span(s):
        if s is None: return "—"
        color = "#f87171" if s > 2 else ("#fbbf24" if s > 1 else ("#34d399" if s < -1 else "#94a3b8"))
        return html.Span(f"{s:+.1f}σ", style={"color": color, "fontWeight": "bold"})

    header = html.Tr([html.Th(h, style={"background": "#1e3a5f", "color": "#93c5fd",
                                         "padding": "4px 6px"})
                      for h in ["Column", "Pod", "Norm. (in/yr)", "σ", "3yr Rate", "σ", "1yr Rate", "10yr Rate"]])
    trows  = []
    for r in rows:
        bg = ("#2d1010" if (r["sig_ns"] or 0) > 2
              else "#241d00" if (r["sig_ns"] or 0) > 1
              else "")
        trows.append(html.Tr([
            html.Td(html.B(r["id"])),
            html.Td(r["pod"]),
            html.Td(_fmt(r["ns"])),
            html.Td(_sig_span(r["sig_ns"])),
            html.Td(_fmt(r["r3"])),
            html.Td(_sig_span(r["sig_r3"])),
            html.Td(_fmt(r["r1"])),
            html.Td(_fmt(r["r10"])),
        ], style={"background": bg, "borderBottom": "1px solid #1e293b"}))

    return html.Table(
        [html.Thead(header), html.Tbody(trows)],
        style={"width": "100%", "borderCollapse": "collapse", "fontSize": "12px"},
    )


@app.callback(
    Output("rate-compare-chart", "figure"),
    Input("scene-data", "data"),
    Input("date-slider", "value"),
    prevent_initial_call=True,
)
def update_rate_compare_chart(data, date_idx):
    if not data:
        raise PreventUpdate
    all_dates     = data["survey_dates"] + data["proj_dates"]
    selected_date = all_dates[date_idx]
    survey_set    = set(data["survey_dates"])

    # Show multi-window rates over time for top 5 columns by current 3yr rate
    sorted_cols = sorted(
        data["columns"],
        key=lambda c: -(c["settlement_rates"].get(selected_date) or 0),
    )[:5]

    fig = go.Figure()
    for col in sorted_cols:
        dates = [d for d in data["survey_dates"] if d in survey_set]
        r3_vals = [col["settlement_rates"].get(d) for d in dates]
        r1_vals = [(col.get("rate_1yr") or {}).get(d) for d in dates]

        fig.add_trace(go.Scatter(
            x=dates, y=r3_vals, mode="lines+markers",
            name=f"{col['id']} 3yr", line=dict(width=2),
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=r1_vals, mode="lines",
            name=f"{col['id']} 1yr", line=dict(width=1, dash="dot"),
            showlegend=True,
        ))

    fig.update_layout(
        title=dict(text="1yr vs 3yr Rate — Top 5 Columns", font=dict(size=11, color="white"), x=0.5),
        xaxis=dict(color="white", gridcolor="#2a2a3e"),
        yaxis=dict(color="white", gridcolor="#2a2a3e", title="Rate (in/yr)"),
        plot_bgcolor="#0d1117", paper_bgcolor="#1e2130",
        font=dict(color="white", size=10),
        legend=dict(font=dict(size=9, color="white"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=10, t=30, b=30),
        height=340,
    )
    return fig


# ---------------------------------------------------------------------------
# Clientside callbacks — Three.js
# ---------------------------------------------------------------------------

app.clientside_callback(
    ClientsideFunction(namespace="settlement3d", function_name="updateScene"),
    Output("three-canvas-trigger", "children"),
    Input("scene-data", "data"),
    Input("date-slider", "value"),
    Input("metric-dropdown", "value"),
    Input("exaggeration-slider", "value"),
    Input("view-mode", "value"),
    Input("color-range-min", "value"),
    Input("color-range-max", "value"),
    Input("plane-toggles", "value"),
    Input("datum-elevation", "value"),
    Input("beam-color-mode", "value"),
    Input("beam-layer-toggles", "value"),
    prevent_initial_call=True,
)

# Dedicated callback for beam layer toggle — updates APP.beamLayers and
# directly sets mesh visibility without waiting for updateScene.
app.clientside_callback(
    ClientsideFunction(namespace="settlement3d", function_name="setBeamLayers"),
    Output("three-dummy", "data"),
    Input("beam-layer-toggles", "value"),
    prevent_initial_call=True,
)

# Relay Three.js click events back to selected-mp store.
# click-poll fires every 400ms so clicks register without needing a slider move.
app.clientside_callback(
    ClientsideFunction(namespace="settlement3d", function_name="getSelectedMP"),
    Output("selected-mp", "data"),
    Input("three-canvas-trigger", "children"),
    Input("click-poll", "n_intervals"),
    prevent_initial_call=True,
)




if __name__ == "__main__":
    app.run(debug=True, port=8050)
