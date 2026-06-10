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

    # ---- controls ----
    CONTROLS_CARD,

    # ---- main panels ----
    dbc.Row([
        # Plan heatmap
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Plan View — Settlement Heatmap", className="py-1 small"),
                dbc.CardBody([
                    dcc.Graph(id="heatmap-chart", style={"height": "480px"},
                              config={"displayModeBar": True, "scrollZoom": True}),
                ], className="p-1"),
            ], style={"background": "#1e2130", "border": "1px solid #2d3250"}),
        ], width=5),
        # Three.js 3D view
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
                        style={"width": "100%", "height": "480px", "position": "relative",
                               "background": "#0d1117"},
                        children=[
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
        ], width=7),
    ], className="mb-2"),

    # ---- sparkline ----
    SPARKLINE_CARD,

    # ---- hidden div needed for clientside output ----
    html.Div(id="three-canvas-trigger", style={"display": "none"}),
], style={"background": "#0d1117", "minHeight": "100vh"})


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
    html_str = _build_report(data, date_idx, metric)
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


# ---------------------------------------------------------------------------
# Report generation helpers
# ---------------------------------------------------------------------------

def _settlement_color(value, max_val):
    """Map settlement value to a hex color (blue→yellow→red)."""
    if value is None or max_val == 0:
        return "#4fc3f7"
    t = min(1.0, value / max(max_val, 0.001))
    if t < 0.5:
        r = int(255 * (t * 2))
        g = int(255 * (0.76 + t * 0.24))
        b = int(255 * (1 - t * 2))
    else:
        r = 255
        g = int(255 * (1 - (t - 0.5) * 2))
        b = 0
    return f"#{r:02x}{g:02x}{b:02x}"


def _build_report(data, date_idx, metric):
    """Generate a self-contained HTML report with embedded Plotly charts."""
    import plotly.io as pio

    all_dates = data["survey_dates"] + data["proj_dates"]
    selected_date = all_dates[date_idx]

    charts_html = []

    # -- Settlement timeline for all columns --
    fig_ts = go.Figure()
    pods = {"A": "#4fc3f7", "B": "#f48fb1"}
    for col in data["columns"]:
        obs_d = [d for d, v in col["settlements"].items() if v is not None]
        obs_v = [col["settlements"][d] for d in obs_d]
        proj_d = [d for d, v in col["proj_settlements"].items() if v is not None]
        proj_v = [col["proj_settlements"][d] for d in proj_d]
        if obs_d and proj_d:
            proj_d = [obs_d[-1]] + proj_d
            proj_v = [obs_v[-1]] + proj_v

        color = pods.get(col["pod"], "gray")
        fig_ts.add_trace(go.Scatter(x=obs_d, y=obs_v, mode="lines",
                                    name=col["id"], line=dict(color=color, width=1.5),
                                    legendgroup=col["pod"],
                                    showlegend=True))
        if proj_d:
            fig_ts.add_trace(go.Scatter(x=proj_d, y=proj_v, mode="lines",
                                        line=dict(color=color, width=1, dash="dot"),
                                        showlegend=False, legendgroup=col["pod"]))

    fig_ts.update_layout(
        title="Cumulative Settlement Over Time",
        xaxis_title="Date", yaxis_title="Settlement (in)",
        plot_bgcolor="#f8f9fa", paper_bgcolor="white",
        height=400,
    )
    charts_html.append(f"<h2>Settlement History</h2>{pio.to_html(fig_ts, full_html=False, include_plotlyjs=False)}")

    # -- Plan heatmap at selected date --
    hm_callback_data = data
    hm_fig = _make_heatmap_for_report(data, selected_date)
    charts_html.append(f"<h2>Settlement Plan View — {selected_date}</h2>{pio.to_html(hm_fig, full_html=False, include_plotlyjs=False)}")

    # -- Summary table --
    rows = []
    for col in data["columns"]:
        last_obs = [v for v in col["settlements"].values() if v is not None]
        last_rate = [v for v in col["settlement_rates"].values() if v is not None]
        rows.append({
            "Point": col["id"],
            "Pod": col["pod"],
            "Max Settlement (in)": f"{max(last_obs):.2f}" if last_obs else "—",
            "Latest Rate (in/yr)": f"{last_rate[-1]:.2f}" if last_rate else "—",
        })
    df_table = pd.DataFrame(rows)
    table_html = df_table.to_html(index=False, border=0,
                                  classes="table table-striped table-sm")

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    report = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SPS Foundation Settlement Report</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
<style>body{{background:#fff; font-family:Arial,sans-serif;}} h2{{color:#2c3e50; margin-top:2rem;}}</style>
</head>
<body class="container-fluid p-4">
<h1>Amundsen-Scott South Pole Station — Foundation Settlement</h1>
<p class="text-muted">Generated {timestamp} | Selected date: {selected_date}</p>
<hr>
<h2>Summary by Monitoring Point</h2>
{table_html}
{"".join(charts_html)}
</body>
</html>"""
    return report


def _make_heatmap_for_report(data, selected_date):
    fig = go.Figure()
    if selected_date in data["heatmap_grids"]:
        gz = np.array(data["heatmap_grids"][selected_date])
        fig.add_trace(go.Heatmap(
            x=data["heatmap_x"], y=data["heatmap_y"], z=gz,
            colorscale="RdBu_r",
            zmin=data["stats"]["min_settlement_in"],
            zmax=data["stats"]["max_settlement_in"],
            colorbar=dict(title="Settlement (in)"),
        ))
    for col in data["columns"]:
        val = col["settlements"].get(selected_date)
        fig.add_trace(go.Scatter(
            x=[col["x"]], y=[col["y"]],
            mode="markers+text",
            marker=dict(size=10, color="white", symbol="square",
                        line=dict(width=1, color="black")),
            text=[col["id"]], textposition="top center", textfont=dict(size=7),
            showlegend=False,
        ))
    fig.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        height=350, margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)
