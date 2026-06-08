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


def _rate_regression_window(settlement_in, skip_dates, window_years=3):
    """Per-MP, per-date rate (in/yr) via regression slope over a trailing window."""
    s = settlement_in.drop(columns=skip_dates, errors="ignore")
    dates = pd.to_datetime(s.columns)
    window_days = window_years * 365.25

    rate_df = pd.DataFrame(np.nan, index=s.index, columns=s.columns, dtype=float)

    for date_str, date_dt in zip(s.columns, dates):
        cutoff = date_dt - pd.Timedelta(days=window_days)
        mask = (dates >= cutoff) & (dates <= date_dt)
        window_cols = s.columns[mask]

        if len(window_cols) < 2:
            continue  # leave as NaN

        ords = pd.to_datetime(window_cols).map(dt.datetime.toordinal).values.astype(float)

        for mp in s.index:
            vals = s.loc[mp, window_cols].values.astype(float)
            ok = ~np.isnan(vals)
            if ok.sum() < 2:
                continue
            slope, *_ = stats.linregress(ords[ok], vals[ok])
            rate_df.loc[mp, date_str] = slope * 365.25  # in/yr

    return rate_df


def _settlement_forecast(settlement_ft, forecast_years, nyears):
    """
    Fit a per-MP linear regression over the trailing `forecast_years` of survey data,
    then project forward `nyears` annual steps.

    Returns
    -------
    proj : DataFrame  — future projected values  (index = proj date strings, cols = MPs)
    reg_line : DataFrame — regression line from window start through last proj date
                           (index = window survey dates + proj date strings, cols = MPs)
    window_start : str — first survey date used in the regression
    """
    s = settlement_ft.drop("2022-01-07", errors="ignore")

    end_dt = pd.to_datetime(s.index[-1])
    cutoff_dt = end_dt - pd.Timedelta(days=forecast_years * 365.25)
    s_window = s.loc[pd.to_datetime(s.index) >= cutoff_dt]
    if len(s_window) < 2:
        s_window = s.iloc[-2:]  # fallback: need at least 2 points

    window_start = s_window.index[0]
    current_year = pd.to_datetime(s_window.index[-1]).year
    proj_dates = [pd.to_datetime(f"{current_year + i + 1}-01-01") for i in range(nyears)]

    win_ords  = pd.to_datetime(s_window.index).map(dt.datetime.toordinal).values.astype(float)
    proj_ords = np.array([dt.datetime.toordinal(d) for d in proj_dates], dtype=float)

    # Combined date list for the regression line: window surveys + proj dates
    reg_dates_dt  = list(pd.to_datetime(s_window.index)) + proj_dates
    reg_ords      = np.array([dt.datetime.toordinal(d) for d in reg_dates_dt], dtype=float)
    reg_dates_str = [d.strftime("%Y-%m-%d") for d in reg_dates_dt]

    proj     = pd.DataFrame(index=[d.strftime("%Y-%m-%d") for d in proj_dates],
                            columns=s_window.columns, dtype=float)
    reg_line = pd.DataFrame(index=reg_dates_str, columns=s_window.columns, dtype=float)

    for col in s_window.columns:
        vals = s_window[col].values.astype(float)
        ok   = ~np.isnan(vals)
        if ok.sum() < 2:
            continue
        slope, intercept, *_ = stats.linregress(win_ords[ok], vals[ok])
        proj[col]     = slope * proj_ords + intercept
        reg_line[col] = slope * reg_ords  + intercept

    return proj.round(4), reg_line.round(4), window_start


def process_all(survey, truss, shim, mp_locations, beams, forecast_years=3, nyears=5):
    """
    Core processing pipeline. Returns a JSON-serialisable dict consumed by
    the Dash app and the Three.js scene.
    """
    shim_ft = shim.div(12)

    # Floor elevation = survey lug + truss offset (confirmed from Dec 2017 onward)
    floor_elev = survey.add(truss).dropna(axis=1, how="all")

    # Grade beam elevation = floor - shim - 12.31
    #   (12.31 ft = spreader beam height + column height to grade beam top)
    # For dates with direct truss+shim measurements, use those exactly.
    # For all other dates (pre-truss surveys, latest surveys, projected dates),
    # approximate with the last-known truss and shim values per MP.
    common_dates = truss.columns.intersection(shim_ft.columns)

    # Per-MP most recent truss and shim values — used wherever a direct measurement
    # is absent (non-floor survey dates, projected dates).  No shim changes assumed.
    truss_last = truss.ffill(axis=1).iloc[:, -1]
    shim_last  = shim_ft.ffill(axis=1).iloc[:, -1]

    # Grade beam elevation = survey + truss - shim - 12.31
    # Seed every survey date with last-known truss/shim, then overwrite with the
    # accurate per-date formula wherever both measurements actually exist.
    grade_beam_elev = survey.add(truss_last, axis=0).sub(shim_last, axis=0).sub(12.31)
    for d in common_dates:
        if d in grade_beam_elev.columns and d in floor_elev.columns:
            grade_beam_elev[d] = floor_elev[d].sub(shim_ft[d]).sub(12.31)
    floor_dates = floor_elev.columns.tolist()

    # Cumulative settlement in inches (positive = settled down)
    # Use each MP's own first non-null reading as its baseline — MPs were instrumented
    # at different times so the first date column is NaN for most points.
    first = survey.apply(lambda row: row.dropna().iloc[0] if not row.dropna().empty else np.nan, axis=1)
    settlement_in = survey.rsub(first, axis=0).mul(12)   # rows=MPs, cols=dates

    # Settlement rate (in/yr) — 3-year trailing regression slope per MP per date
    # Excludes erroneous / duplicate surveys before fitting
    _skip = ["2022-01-07", "2010-11-03"]  # 2010-11-03 is 1-day duplicate of 2010-11-02
    rate_in_yr = _rate_regression_window(settlement_in, _skip, window_years=3)

    # Forecast (uses ft for regression, convert result back to inches)
    settlement_ft_T = settlement_in.div(12)   # MP x dates (ft)
    proj_ft, reg_line_ft, forecast_window_start = _settlement_forecast(
        settlement_ft_T.T, forecast_years, nyears)
    proj_ft      = proj_ft.T       # MP x proj_dates
    reg_line_ft  = reg_line_ft.T   # MP x (window survey dates + proj dates)
    proj_in      = proj_ft.mul(12)
    reg_line_in  = reg_line_ft.mul(12)

    all_dates = survey.columns.tolist()
    proj_dates = proj_in.columns.tolist()

    # Extend floor_elev to non-floor survey dates so beam diffs cover every date.
    # grade_beam_elev already covers all survey dates from the seed above.
    non_floor = [d for d in all_dates if d not in floor_dates]
    if non_floor:
        floor_elev = pd.concat(
            [floor_elev, survey[non_floor].add(truss_last, axis=0)], axis=1
        ).sort_index(axis=1)

    # Extend both to projected dates using the projected survey lug elevation.
    # proj_survey = first_lug - projected_settlement  →  floor = proj_survey + truss_last
    if proj_dates:
        proj_survey_ft = proj_ft.rsub(first, axis=0)
        proj_floor_df  = proj_survey_ft.add(truss_last, axis=0)
        proj_gb_df     = proj_floor_df.sub(shim_last, axis=0).sub(12.31)
        floor_elev      = pd.concat([floor_elev, proj_floor_df], axis=1)
        grade_beam_elev = pd.concat([grade_beam_elev, proj_gb_df], axis=1)

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
        fl_series = reg_line_in.loc[mp] if mp in reg_line_in.index else pd.Series(dtype=float)
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
            "forecast_line": _series_to_dict(fl_series),
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

    # Rate mean/std at the latest survey — used for ±1 STD default color range
    latest_rate_date = rate_in_yr.columns[-1] if len(rate_in_yr.columns) else None
    if latest_rate_date is not None:
        latest_rates = [float(rate_in_yr.loc[m, latest_rate_date])
                        for m in rate_in_yr.index
                        if pd.notna(rate_in_yr.loc[m, latest_rate_date])]
    else:
        latest_rates = []
    rate_mean = float(np.mean(latest_rates)) if latest_rates else 0.0
    rate_std  = float(np.std(latest_rates))  if latest_rates else 0.0

    # Fixed datum: 0.5 ft below the minimum grade beam elevation at the latest floor survey.
    # Using the latest (most-settled) state ensures all columns sit above the reference
    # plane regardless of how much they've settled since installation.
    datum_gb = 0.0
    if floor_dates:
        latest_floor = floor_dates[-1]
        gb_vals = [float(grade_beam_elev.loc[m, latest_floor])
                   for m in mp_locations.index
                   if m in grade_beam_elev.index and pd.notna(grade_beam_elev.loc[m, latest_floor])]
        datum_gb = round(float(min(gb_vals)) - 0.5, 2) if gb_vals else 0.0

    return {
        "columns": columns_out,
        "beams": beams_out,
        "survey_dates": all_dates,
        "floor_dates": floor_dates,
        "proj_dates": proj_dates,
        "forecast_window_start": forecast_window_start,
        "heatmap_grids": heatmap_grids,
        "heatmap_x": gx.tolist(),
        "heatmap_y": gy.tolist(),
        "stats": {
            "max_settlement_in": float(max(all_s)) if all_s else 0.0,
            "min_settlement_in": float(min(all_s)) if all_s else 0.0,
            "max_rate_in_yr": float(max(all_r)) if all_r else 0.0,
            "rate_mean": round(rate_mean, 4),
            "rate_std":  round(rate_std,  4),
            "datum_grade_beam": datum_gb,
        },
    }
