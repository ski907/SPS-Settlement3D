import os
import datetime as dt
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.interpolate import griddata


def find_beam_csv():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in [
        os.path.join(script_dir, "SP_BeamArrowLabels.csv"),
        os.path.join(script_dir, "..", "SP_BeamArrowLabels.csv"),
    ]:
        if os.path.exists(candidate):
            return os.path.normpath(candidate)
    raise FileNotFoundError("SP_BeamArrowLabels.csv not found next to app or in parent directory")


def load_beam_info():
    csv_path = find_beam_csv()
    df = pd.read_csv(csv_path)

    mp_locations = (df[["MP_W_S", "mpX", "mpY"]]
                    .rename(columns={"MP_W_S": "MONITOR_POINT"})
                    .dropna()
                    .set_index("MONITOR_POINT"))

    beams = df[["MP_W_S", "MP_E_N", "beamName", "beamLength",
                "startX", "startY", "endX", "endY"]].dropna(subset=["beamName", "beamLength"])

    return mp_locations, beams


def read_excel_sheet(xl_file, sheet):
    if sheet == "SURVEY DATA":
        data = pd.read_excel(xl_file, engine="openpyxl", sheet_name="SURVEY DATA",
                             skiprows=[0, 2, 3], nrows=36)
        data = (data.dropna(axis=1, how="all")
                .drop(columns=["DESCRIPTION", "Shims\nNote 13", "Delta"])
                .rename(columns={"MONITOR\nPOINT": "MONITOR_POINT",
                                 "2010-11-02 00:00:00.1": "2010-11-03 00:00:00"})
                .set_index("MONITOR_POINT"))
    else:
        data = pd.read_excel(xl_file, engine="openpyxl", sheet_name=sheet,
                             skiprows=[0, 2, 3], nrows=36)
        data = (data.dropna(axis=1, how="all")
                .drop(columns=["DESCRIPTION", "Shims", "Delta"])
                .rename(columns={"MONITOR\nPOINT": "MONITOR_POINT"})
                .set_index("MONITOR_POINT"))

    data.columns = pd.to_datetime(data.columns).strftime("%Y-%m-%d")
    return data


def _settlement_forecast(settlement_ft, nsurvey, nyears):
    s = settlement_ft.drop("2022-01-07", errors="ignore")
    s_window = s.iloc[-nsurvey:]
    current_year = pd.to_datetime(s_window.index[-1]).year

    proj_dates = [pd.to_datetime(f"{current_year + i + 1}-01-01") for i in range(nyears)]

    s_ord = s_window.copy()
    s_ord.index = pd.to_datetime(s_ord.index).map(dt.datetime.toordinal)
    proj_ord = [dt.datetime.toordinal(d) for d in proj_dates]
    delta_days = [o - s_ord.index[-1] for o in proj_ord]
    starting = s_window.iloc[-1]

    reg = s_ord.apply(lambda x: stats.linregress(s_ord.index, x), result_type="expand").rename(
        index={0: "slope", 1: "intercept", 2: "r", 3: "p", 4: "se"})

    proj = pd.DataFrame(index=proj_dates, columns=s_window.columns, dtype=float)
    for col in reg.columns:
        slope = reg.loc["slope", col]
        proj[col] = [starting[col] + slope * d for d in delta_days]

    proj.index = proj.index.strftime("%Y-%m-%d")
    return proj.round(4)


def process_all(survey, truss, shim, mp_locations, beams, nsurvey=6, nyears=5):
    """
    Core processing pipeline. Returns a JSON-serialisable dict consumed by
    the Dash app and the Three.js scene.
    """
    shim_ft = shim.div(12)

    # Floor elevation = survey lug + truss offset (confirmed from Dec 2017 onward)
    floor_elev = survey.add(truss).dropna(axis=1, how="all")

    # Grade beam elevation:
    #   For dates where both truss and shim data exist (Dec 2017+):
    #     grade_beam = floor_elev - shim_pack - 12.31 ft constant
    #     (12.31 ft = spreader beam height + column height above shim support to grade beam top)
    #   For pre-2017 dates: use a fixed offset derived from the earliest available reference
    #   so that relative Z positioning still tracks the survey lug correctly.
    common_dates = truss.columns.intersection(shim_ft.columns)
    ref_col = "2017-12-01" if "2017-12-01" in truss.columns else (
        truss.columns[0] if len(truss.columns) > 0 else None)

    if ref_col:
        # Best-estimate offset for pre-reference dates (keeps relative positions consistent)
        offset_ref = shim_ft[ref_col].add(12.31).sub(truss[ref_col])
        grade_beam_elev = survey.sub(offset_ref, axis=0)
        # Overwrite with accurate per-date values where truss+shim data exists
        for d in common_dates:
            if d in grade_beam_elev.columns and d in floor_elev.columns:
                grade_beam_elev[d] = floor_elev[d].sub(shim_ft[d]).sub(12.31)
    else:
        grade_beam_elev = survey.copy()
    floor_dates = floor_elev.columns.tolist()

    # Cumulative settlement in inches (positive = settled down)
    # Use each MP's own first non-null reading as its baseline — MPs were instrumented
    # at different times so the first date column is NaN for most points.
    first = survey.apply(lambda row: row.dropna().iloc[0] if not row.dropna().empty else np.nan, axis=1)
    settlement_in = survey.rsub(first, axis=0).mul(12)   # rows=MPs, cols=dates

    # Inter-survey settlement rate (in/yr) — exclude erroneous / duplicate surveys
    _skip = ["2022-01-07", "2010-11-03"]  # 2010-11-03 is 1-day duplicate of 2010-11-02
    s_no2022 = settlement_in.drop(columns=_skip, errors="ignore")
    delta_in = s_no2022.diff(axis=1)
    diff_days = pd.Series(
        pd.to_datetime(s_no2022.columns).to_series().diff().dt.days.values,
        index=s_no2022.columns,
        dtype=float,
    )
    rate_in_yr = delta_in.div(diff_days, axis=1).mul(365)

    # Forecast (uses ft for regression, convert result back to inches)
    settlement_ft_T = settlement_in.div(12)   # MP x dates (ft)
    proj_ft = _settlement_forecast(settlement_ft_T.T, nsurvey, nyears).T  # MP x proj_dates
    proj_in = proj_ft.mul(12)

    all_dates = survey.columns.tolist()
    proj_dates = proj_in.columns.tolist()

    # --- Build per-column records ---
    columns_out = []
    for mp in mp_locations.index:
        x = float(mp_locations.loc[mp, "mpX"])
        y = float(mp_locations.loc[mp, "mpY"])

        def _series_to_dict(series):
            return {d: (float(v) if pd.notna(v) else None)
                    for d, v in series.items()}

        s_series  = settlement_in.loc[mp] if mp in settlement_in.index else pd.Series(dtype=float)
        fe_series = floor_elev.loc[mp] if mp in floor_elev.index else pd.Series(dtype=float)
        gb_series = grade_beam_elev.loc[mp] if mp in grade_beam_elev.index else pd.Series(dtype=float)
        r_series  = rate_in_yr.loc[mp] if mp in rate_in_yr.index else pd.Series(dtype=float)
        p_series  = proj_in.loc[mp] if mp in proj_in.index else pd.Series(dtype=float)
        sh_series = shim.loc[mp] if mp in shim.index else pd.Series(dtype=float)

        columns_out.append({
            "id": mp,
            "x": x,
            "y": y,
            "pod": mp[0],
            "settlements": _series_to_dict(s_series),
            "floor_elevations": _series_to_dict(fe_series),
            "grade_beam_elevations": _series_to_dict(gb_series),
            "settlement_rates": _series_to_dict(r_series),
            "proj_settlements": _series_to_dict(p_series),
            "shim_inches": _series_to_dict(sh_series),
        })

    # --- Build per-beam records ---
    # Connections between pods A and B at these two locations are a different structural
    # element type (not a standard rigid floor/grade beam) and are rendered distinctly.
    _INTER_POD = {frozenset(["A3-3", "B2-1"]), frozenset(["A3-4", "B2-4"])}

    beams_out = []
    for _, row in beams.iterrows():
        sid, eid = row["MP_W_S"], row["MP_E_N"]

        def _diff_series(df, a, b):
            if a in df.index and b in df.index:
                return {d: (float(abs(df.loc[a, d] - df.loc[b, d]) * 12)
                            if pd.notna(df.loc[a, d]) and pd.notna(df.loc[b, d]) else None)
                        for d in df.columns}
            return {}

        beams_out.append({
            "id": row["beamName"],
            "start_id": sid,
            "end_id": eid,
            "start_x": float(mp_locations.loc[sid, "mpX"]) if sid in mp_locations.index else None,
            "start_y": float(mp_locations.loc[sid, "mpY"]) if sid in mp_locations.index else None,
            "end_x": float(mp_locations.loc[eid, "mpX"]) if eid in mp_locations.index else None,
            "end_y": float(mp_locations.loc[eid, "mpY"]) if eid in mp_locations.index else None,
            "length_ft": float(row["beamLength"]),
            "floor_diffs": _diff_series(floor_elev, sid, eid),
            "grade_beam_diffs": _diff_series(grade_beam_elev, sid, eid),
            "is_inter_pod": bool(frozenset([sid, eid]) in _INTER_POD),
        })

    # --- Interpolated heatmap grids ---
    mp_xy = np.array([[float(mp_locations.loc[m, "mpX"]), float(mp_locations.loc[m, "mpY"])]
                      for m in mp_locations.index if m in settlement_in.index])
    mp_ids = [m for m in mp_locations.index if m in settlement_in.index]

    gx = np.linspace(0, 400, 120)
    gy = np.linspace(0, 130, 50)
    GX, GY = np.meshgrid(gx, gy)

    heatmap_grids = {}
    for date in all_dates:
        vals = np.array([float(settlement_in.loc[m, date]) if pd.notna(settlement_in.loc[m, date]) else np.nan
                         for m in mp_ids])
        if not np.all(np.isnan(vals)):
            gz = griddata(mp_xy, vals, (GX, GY), method="cubic")
            nan_mask = np.isnan(gz)
            if nan_mask.any():
                gz[nan_mask] = griddata(mp_xy, vals, (GX[nan_mask], GY[nan_mask]), method="nearest")
            heatmap_grids[date] = gz.tolist()

    # Statistics for color scale anchoring
    all_s = [v for col in columns_out for v in col["settlements"].values() if v is not None]
    all_r = [v for col in columns_out for v in col["settlement_rates"].values() if v is not None and not np.isnan(v)]

    # Fixed datum: mean grade beam elevation at the earliest floor survey date
    datum_gb = 0.0
    if floor_dates:
        earliest_floor = floor_dates[0]
        gb_vals = [float(grade_beam_elev.loc[m, earliest_floor])
                   for m in mp_locations.index
                   if m in grade_beam_elev.index and pd.notna(grade_beam_elev.loc[m, earliest_floor])]
        datum_gb = round(float(np.mean(gb_vals)), 2) if gb_vals else 0.0

    return {
        "columns": columns_out,
        "beams": beams_out,
        "survey_dates": all_dates,
        "floor_dates": floor_dates,
        "proj_dates": proj_dates,
        "heatmap_grids": heatmap_grids,
        "heatmap_x": gx.tolist(),
        "heatmap_y": gy.tolist(),
        "stats": {
            "max_settlement_in": float(max(all_s)) if all_s else 0.0,
            "min_settlement_in": float(min(all_s)) if all_s else 0.0,
            "max_rate_in_yr": float(max(all_r)) if all_r else 0.0,
            "datum_grade_beam": datum_gb,
        },
    }
