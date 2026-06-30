"""
stress_calcs.py  —  Grade-beam bending stress via central-difference curvature.

Method
------
The grade-beam elevation at each column is a discrete sample of the foundation's
deflection profile.  Where three columns sit on the same grade-beam line
(connected by beams that share a common direction), the curvature κ at the
middle column is approximated by the non-uniform-spacing central-difference
formula:

          2 [ z_prev · h_next  +  z_next · h_prev  −  z_node · (h_prev + h_next) ]
κ_node ≈  ─────────────────────────────────────────────────────────────────────────
                          h_prev · h_next · (h_prev + h_next)

    z  = grade-beam elevation at a column, converted to inches
    h  = centre-to-centre distance between adjacent columns on the line, inches

κ is dimensionally 1/in (curvature of the deflection curve).

Bending stress at that node (elastic beam theory):
    σ = E · κ · c        (ksi)

where:
    E   = 29 000 ksi   Young's modulus for steel
    c   = 14 in        distance from neutral axis to extreme fibre
    F_y = 50 ksi       yield stress (used for colour-scale normalisation only)

How curvature is assigned
--------------------------
κ is a *node* quantity — it describes how sharply the settlement profile bends
at a particular column, computed from a collinear triplet of three columns on
the same beam line.

The collinearity search uses the beam's own unit direction vector, so a
neighbour connected at a corner or T-junction (perpendicular direction) cannot
satisfy the threshold and is never accepted as a triplet partner.  End-of-run
nodes — where the beam terminates with no collinear continuation — have no
valid triplet from that side and contribute no curvature.

Outputs
-------
compute_grade_beam_stress() returns a tuple:

  beam_stress : dict  beam_name → {date: {"s": ksi_or_None, "e": ksi_or_None}}
      Per-beam start / end node stresses for colour-coding grade beams.
      "s" = stress at the start (S) endpoint; "e" = stress at the end (E)
      endpoint.  None where no valid collinear triplet exists at that end.

  col_curv : dict  col_id → {date: [{prev, next, k6, ksi}, ...]}
      Per-column curvature results for display in column tooltips / reports.
      Each list entry is one valid triplet where that column is the middle node:
        prev  — column ID on one side
        next  — column ID on the other side
        k6    — κ in units of 10⁻⁶ /in   (display-friendly: yield ≈ 123)
        ksi   — bending stress in ksi
"""

import numpy as np
import pandas as pd
from collections import defaultdict

# ── Material / section constants ──────────────────────────────────────────────
E_KSI  = 29_000.0   # Young's modulus, ksi
C_IN   = 14.0       # Extreme-fibre distance, in
FY_KSI = 50.0       # Yield stress, ksi

# Minimum cosine of the angle between the beam direction and a candidate
# extension to be accepted as collinear.  cos(20°) ≈ 0.94 rejects perpendicular
# beams at corners while tolerating minor misalignment in the grid.
_COLLINEAR_COS = 0.94


# ── Internal helpers ──────────────────────────────────────────────────────────

def _col_pos(mp_locations):
    """Return {mp_id: np.array([x_ft, y_ft])} for every known column."""
    return {
        mp: np.array([float(mp_locations.loc[mp, "mpX"]),
                      float(mp_locations.loc[mp, "mpY"])])
        for mp in mp_locations.index
    }


def _collinear_extension(node_id, direction, beam_adj, pos):
    """
    Find the neighbouring column that lies along `direction` from `node_id`,
    connected by a grade beam, and collinear with that direction.

    Parameters
    ----------
    node_id   : column to search from
    direction : unit vector (np.array shape (2,)) pointing in the search direction
    beam_adj  : {node_id: [neighbour_id, ...]} — grade-beam adjacency
    pos       : {node_id: np.array([x, y])} — column positions in feet

    Returns
    -------
    (neighbour_id, distance_in) or (None, None) if no collinear neighbour found
    """
    best_id, best_cos = None, _COLLINEAR_COS
    for nbr in beam_adj[node_id]:
        vec = pos[nbr] - pos[node_id]
        dist_ft = float(np.linalg.norm(vec))
        if dist_ft < 1e-6:
            continue
        cos_theta = float(np.dot(direction, vec / dist_ft))
        if cos_theta > best_cos:
            best_cos = cos_theta
            best_id = nbr
    if best_id is None:
        return None, None
    dist_in = float(np.linalg.norm(pos[best_id] - pos[node_id])) * 12.0
    return best_id, dist_in


# ── Public API ────────────────────────────────────────────────────────────────

def compute_grade_beam_stress(beams_df, grade_beam_elev, mp_locations):
    """
    Compute bending stress for every grade beam at every date.

    Parameters
    ----------
    beams_df        : DataFrame from load_beam_info() — one row per beam,
                      columns include MP_W_S, MP_E_N, beamName
    grade_beam_elev : DataFrame, index = column IDs, columns = date strings,
                      values = grade-beam elevation in feet
    mp_locations    : DataFrame, index = column IDs,
                      with 'mpX' and 'mpY' columns in feet

    Returns
    -------
    (beam_stress, col_curv)

    beam_stress : dict  beam_name → {date: {"s": ksi_or_None, "e": ksi_or_None}}
    col_curv    : dict  col_id → {date: [{prev, next, k6, ksi}, ...]}
    """
    pos      = _col_pos(mp_locations)
    all_dates = grade_beam_elev.columns.tolist()

    # Grade-beam adjacency
    beam_adj = defaultdict(list)
    for _, row in beams_df.iterrows():
        sid, eid = row["MP_W_S"], row["MP_E_N"]
        if sid in pos and eid in pos:
            beam_adj[sid].append(eid)
            beam_adj[eid].append(sid)

    def get_elev_in(col_id, date):
        if col_id not in grade_beam_elev.index:
            return None
        val = grade_beam_elev.loc[col_id, date] if date in grade_beam_elev.columns else np.nan
        return float(val) * 12.0 if pd.notna(val) else None

    # ── Pass 1: find all unique collinear triplets ────────────────────────────
    # For each beam S→E, check for a collinear extension behind S and one past E.
    # Triplets are stored per middle-column; deduplication prevents double-counting
    # the same triplet reached from two different beams.
    #
    # triplets_by_col[col_id] = list of {"prev", "next", "h1", "h2"}
    #   prev / next — column IDs on either side of this middle node
    #   h1 / h2     — distances in inches (prev→node and node→next)

    triplets_by_col = defaultdict(list)
    seen_by_col     = defaultdict(set)   # col_id → set of frozenset({prev, next})

    # Also record per-beam geometry for Pass 3
    beam_geom = {}   # beam_id → {"sid", "eid", "prev_id", "next_id"}

    for _, row in beams_df.iterrows():
        sid, eid = row["MP_W_S"], row["MP_E_N"]
        bid      = row["beamName"]

        if sid not in pos or eid not in pos:
            beam_geom[bid] = None
            continue

        vec_se  = pos[eid] - pos[sid]
        span_ft = float(np.linalg.norm(vec_se))
        if span_ft < 1e-6:
            beam_geom[bid] = None
            continue

        dir_se = vec_se / span_ft
        h_se   = span_ft * 12.0

        prev_id, h_ps = _collinear_extension(sid, -dir_se, beam_adj, pos)
        next_id, h_en = _collinear_extension(eid,  dir_se, beam_adj, pos)

        beam_geom[bid] = {
            "sid": sid, "eid": eid,
            "prev_id": prev_id, "next_id": next_id,
        }

        # Triplet at the START node: (prev_id → sid → eid)
        if prev_id is not None:
            key = frozenset({prev_id, eid})
            if key not in seen_by_col[sid]:
                seen_by_col[sid].add(key)
                triplets_by_col[sid].append(
                    {"prev": prev_id, "next": eid, "h1": h_ps, "h2": h_se}
                )

        # Triplet at the END node: (sid → eid → next_id)
        if next_id is not None:
            key = frozenset({sid, next_id})
            if key not in seen_by_col[eid]:
                seen_by_col[eid].add(key)
                triplets_by_col[eid].append(
                    {"prev": sid, "next": next_id, "h1": h_se, "h2": h_en}
                )

    # ── Pass 2: compute per-date curvature for each triplet ───────────────────
    # col_curv[col_id][date] = [{prev, next, k6, ksi}, ...]

    col_curv = defaultdict(lambda: defaultdict(list))

    for col_id, col_trips in triplets_by_col.items():
        for t in col_trips:
            prev_id, next_id = t["prev"], t["next"]
            h1, h2           = t["h1"],   t["h2"]
            for date in all_dates:
                z_p = get_elev_in(prev_id, date)
                z_c = get_elev_in(col_id,  date)
                z_n = get_elev_in(next_id, date)
                if z_p is None or z_c is None or z_n is None:
                    continue
                k = abs(2.0 * (z_p * h2 + z_n * h1 - z_c * (h1 + h2))
                        / (h1 * h2 * (h1 + h2)))
                col_curv[col_id][date].append({
                    "prev": prev_id,
                    "next": next_id,
                    "k6":  round(k * 1e6, 3),
                    "ksi": round(E_KSI * k * C_IN, 3),
                })

    # ── Pass 3: derive per-beam start / end stresses from col_curv ───────────
    # For beam S→E:
    #   stress at S = entry in col_curv[S] where "next" == E
    #   stress at E = entry in col_curv[E] where "prev" == S

    beam_stress = {}

    for bid, geom in beam_geom.items():
        if geom is None:
            beam_stress[bid] = {}
            continue

        sid, eid = geom["sid"], geom["eid"]
        by_date  = {}

        for date in all_dates:
            s_ksi = next(
                (e["ksi"] for e in col_curv[sid].get(date, []) if e["next"] == eid),
                None,
            )
            e_ksi = next(
                (e["ksi"] for e in col_curv[eid].get(date, []) if e["prev"] == sid),
                None,
            )
            if s_ksi is not None or e_ksi is not None:
                by_date[date] = {"s": s_ksi, "e": e_ksi}
            else:
                by_date[date] = None

        beam_stress[bid] = by_date

    # Convert nested defaultdicts to plain dicts for JSON serialisation
    col_curv_out = {
        col_id: {date: entries for date, entries in dates.items()}
        for col_id, dates in col_curv.items()
    }

    return beam_stress, col_curv_out
