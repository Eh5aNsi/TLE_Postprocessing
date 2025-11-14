"""
***Target Localization Error(TLE) Analysis***

Outputs per sheet:
  tle_outputs/<SheetName>/
    - results_table_<SheetName>.csv
    - TLE_bars_two_trials_<SheetName>.png
    - pin_axes_3d_<SheetName>.png
    - pin_axes_panels_<SheetName>.png

Master outputs:
  tle_outputs/results_all_users.csv
  tle_outputs/summary_by_user.csv
  
  ==============================================================
  
  Author: [Ehsan Nasiri]
  Date: Oct. 2025 --> modified Nov. 11
  
  Email:[Ehsan.Nasiri@dartmouth.edu]
  
"""

import argparse
import os
import math
import warnings
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# -------------------- Config  --------------------
DEFAULT_WORKBOOK = "TLE Data.xlsx"  # chnage the file name if necessary!!
STUDY_NAME = "Throat Model Target Localization" #change the file name if necessary...
BEAD_DIAMETER_MM = 2.381  # mm
YMAX_LEFT = 9             # mm (bars axis)
YMAX_RIGHT = 180          # deg (AE axis)
GUIDE1 = 3                # mm
GUIDE2 = 5                # mm

C_NO  = np.array([0.0000, 0.4470, 0.7410])  # blue
C_NAV = np.array([0.2000, 0.6000, 0.2000])  # green

ROOT_OUTDIR = "TLE_Outputs"
PIVOT_BACKOFF_MM = 37.56

# -------------------- Utilities --------------------

def safename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))

def to_num(x) -> float:
    if isinstance(x, (int, float, np.integer, np.floating)) and not pd.isna(x):
        return float(x)
    try:
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def find_all_titles(df: pd.DataFrame, title: str) -> List[Tuple[int, int]]:
    hits = []
    vals = df.values
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if isinstance(vals[i, j], str) and vals[i, j] == title:
                hits.append((i, j))
    return hits

def match_row(positions: List[Tuple[int, int]], row_wanted: int) -> Optional[Tuple[int, int]]:
    for r, c in positions:
        if r == row_wanted:
            return (r, c)
    return None

def parse_side_by_side_table(df: pd.DataFrame, r_title: int, c_title: int, has_no_col: bool) -> pd.DataFrame:
    """
      data starts at r = r_title + 2
      L, P, S columns start at c_title + (1 if has_no_col else 0)
      No column at c_title iff has_no_col
      Stop after we see >=3 consecutive empty L/P/S rows *after* data started.
    Returns DataFrame with columns: No, L, P, S .
    """
    r = r_title + 2
    c0 = c_title
    Lcol = c0 + (1 if has_no_col else 0)
    Pcol = Lcol + 1
    Scol = Lcol + 2
    Ncol = c0

    rows = []
    seen_data = False
    empty_streak = 0

    nrows, ncols = df.shape
    while r < nrows:
        L = to_num(df.iat[r, Lcol]) if Lcol < ncols else float("nan")
        P = to_num(df.iat[r, Pcol]) if Pcol < ncols else float("nan")
        S = to_num(df.iat[r, Scol]) if Scol < ncols else float("nan")
        No = to_num(df.iat[r, Ncol]) if (has_no_col and Ncol < ncols) else float("nan")

        if all(math.isnan(v) for v in (L, P, S)):
            if seen_data:
                empty_streak += 1
                if empty_streak >= 3:
                    break
            r += 1
            continue

        seen_data = True
        empty_streak = 0
        rows.append((No, L, P, S))
        r += 1

    if not rows:
        return pd.DataFrame(columns=["No", "L", "P", "S"])

    T = pd.DataFrame(rows, columns=["No", "L", "P", "S"])

    if has_no_col:
        if T["No"].isna().all():
            T["No"] = np.arange(1, len(T) + 1, dtype=float)
        else:
            # Forward-fill "No" 
            T["No"] = T["No"].ffill()
            if pd.isna(T["No"].iloc[0]):
                T.loc[T.index[0], "No"] = 1.0
    else:
        T["No"] = np.arange(1, len(T) + 1, dtype=float)

    # Keep only numeric columns and cast
    for col in ["No", "L", "P", "S"]:
        T[col] = pd.to_numeric(T[col], errors="coerce")

    # Deduplicate by No
    _, idx = np.unique(T["No"].values, return_index=True)
    T = T.iloc[np.sort(idx)].reset_index(drop=True)
    return T

def row_LPS_to_vec(tbl: pd.DataFrame, no: float) -> np.ndarray:
    row = tbl.loc[tbl["No"] == no]
    if row.empty:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    row = row.iloc[0]
    return np.array([row["L"], row["P"], row["S"]], dtype=float)

def best_fit_direction(pts3x3: np.ndarray) -> np.ndarray:
    mu = pts3x3.mean(axis=0)
    X = pts3x3 - mu
    # econ SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    d = Vt[0, :]  # principal component
    # orient
    if np.dot(d, (pts3x3[2, :] - pts3x3[0, :])) < 0:
        d = -d
    nrm = np.linalg.norm(d)
    if nrm == 0:
        return np.array([1.0, 0.0, 0.0])
    return d / nrm

# def angle_between(v: np.ndarray, w: np.ndarray, degrees: bool = True) -> float:
#     v = v / max(np.linalg.norm(v), np.finfo(float).eps)
#     w = w / max(np.linalg.norm(w), np.finfo(float).eps)
#     c = float(np.clip(np.dot(v, w), -1.0, 1.0))
#     ang = math.acos(c)
#     return math.degrees(ang) if degrees else ang

def _angle_between(u, v):
    u = u / max(np.linalg.norm(u), np.finfo(float).eps)
    v = v / max(np.linalg.norm(v), np.finfo(float).eps)
    c = float(np.clip(np.abs(np.dot(u, v)), -1.0, 1.0))  
    return math.degrees(math.acos(c))

def ranksum_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    if _HAVE_SCIPY:
        try:
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            return float(p)
        except Exception:
            return float("nan")
    return float("nan")

def mean_safe(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.mean(x)) if x.size else float("nan")

def std_safe(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.std(x, ddof=0)) if x.size else float("nan")

def set_axes_equal_3d(ax, pts: np.ndarray):
    """Set 3D axes equal based on point cloud bounds."""
    if pts.size == 0:
        return
    xmin, ymin, zmin = pts.min(axis=0)
    xmax, ymax, zmax = pts.max(axis=0)
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    zmid = 0.5 * (zmin + zmax)
    max_range = 0.5 * max(xmax - xmin, ymax - ymin, zmax - zmin, 1e-9)
    ax.set_xlim(xmid - max_range, xmid + max_range)
    ax.set_ylim(ymid - max_range, ymid + max_range)
    ax.set_zlim(zmid - max_range, zmid + max_range)

# -------------------- per-sheet analysis --------------------
def analyze_sheet(sheet_name: str, workbook_path: str, outdir: str):
    """
    Returns:
        T (pd.DataFrame): per-pin results with columns
            User, No, group, TLE_center, TLE_surface, TLEx, TLEy, AE_deg
        stats: summary per user....
    """
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    # Read raw cells 
    df = pd.read_excel(workbook_path, sheet_name=sheet_name, header=None, engine=None)

    pos_bead = find_all_titles(df, "Bead locations Postop")
    pos_tip  = find_all_titles(df, "Pin tip locations")
    pos_ap1  = find_all_titles(df, "Additional pin point #1")
    pos_ap2  = find_all_titles(df, "Additional pin point #2")
    if not pos_bead or not pos_tip or not pos_ap1 or not pos_ap2:
        raise RuntimeError(f'Could not find all four titles in sheet "{sheet_name}".')

    bead_rows = sorted(set(r for r, _ in pos_bead))
    blocks = []
    for r in bead_rows:
        t  = match_row(pos_tip,  r)
        a1 = match_row(pos_ap1,  r)
        a2 = match_row(pos_ap2,  r)
        if t and a1 and a2:
            cB = [c for rr, c in pos_bead if rr == r][0]
            blocks.append((r, cB, t[1], a1[1], a2[1]))

    if len(blocks) < 2:
        raise RuntimeError(f'Expected two complete blocks (Non-Nav & Nav) on "{sheet_name}", found {len(blocks)}.')

    blocks.sort(key=lambda x: x[0])
    blocks = blocks[:2]  # top two blocks
    labels = ["No Navigation", "With Navigation"]
    colors = [C_NO, C_NAV]

    all_B, all_P1, all_P2, all_P3, all_No = [], [], [], [], []

    # Parse both trials
    for b in range(2):
        r, cB, cT, c1, c2 = blocks[b]
        bead_blk = parse_side_by_side_table(df, r, cB, has_no_col=True)
        tip_blk  = parse_side_by_side_table(df, r, cT, has_no_col=True)
        ap1_blk  = parse_side_by_side_table(df, r, c1, has_no_col=False)
        ap2_blk  = parse_side_by_side_table(df, r, c2, has_no_col=False)

        if bead_blk.empty or tip_blk.empty or ap1_blk.empty or ap2_blk.empty:
            raise RuntimeError(f'Block {b+1} incomplete on "{sheet_name}".')

        # Overlap pin No's
        nos = sorted(set(bead_blk["No"]).intersection(set(tip_blk["No"])))
        if not nos:
            raise RuntimeError(f'Block {b+1} has no overlapping "No" between Bead and Tip on "{sheet_name}".')
        if len(ap1_blk) < max(nos) or len(ap2_blk) < max(nos):
            raise RuntimeError(f'Block {b+1}: AP tables too short for max(No)={int(max(nos))}.')

        n = len(nos)
        B   = np.zeros((n, 3), float)
        P1s = np.zeros((n, 3), float)
        P2s = np.zeros((n, 3), float)
        P3s = np.zeros((n, 3), float)
        for i, no in enumerate(nos):
            B[i, :]   = row_LPS_to_vec(bead_blk, no)
            P1s[i, :] = row_LPS_to_vec(tip_blk,  no)
            # AP tables indexed by row number (no)
            a1 = ap1_blk.loc[ap1_blk.index == int(no) - 1, ["L", "P", "S"]].values
            a2 = ap2_blk.loc[ap2_blk.index == int(no) - 1, ["L", "P", "S"]].values
            if a1.size == 0 or a2.size == 0:
                raise RuntimeError(f'AP rows missing for pin No={int(no)} on "{sheet_name}".')
            P2s[i, :] = a1.flatten().astype(float)
            P3s[i, :] = a2.flatten().astype(float)

        all_B.append(B); all_P1.append(P1s); all_P2.append(P2s); all_P3.append(P3s); all_No.append(np.array(nos, int))

    # Compute metrics for both trials
    results_rows = []
    TLE_group = [{"no": None, "ae": None}, {"no": None, "ae": None}]
    bead_r = BEAD_DIAMETER_MM / 2.0

    for b in range(2):
        B   = all_B[b];  P1s = all_P1[b];  P2s = all_P2[b];  P3s = all_P3[b];  nos = all_No[b]
        n = B.shape[0]
        TLE_center = np.zeros(n, float)
        TLEx = np.zeros(n, float)
        TLEy = np.zeros(n, float)
        AEdeg = np.zeros(n, float)

        for k in range(n):
            P1 = P1s[k, :]; P2 = P2s[k, :]; P3 = P3s[k, :]; Bb = B[k, :]
            d  = best_fit_direction(np.vstack([P1, P2, P3]))
            # --- TLE (TIP-based) ---
            v3  = Bb - P1                   # tip → bead
            v3x = np.dot(v3, d) * d
            v3y = v3 - v3x

            TLE_center[k] = np.linalg.norm(v3)
            TLEx[k]       = abs(np.dot(v3, d))
            TLEy[k]       = np.linalg.norm(v3y)
            
            s = 1.0 if np.dot(P3 - P1, d) >= 0 else -1.0
            W = P1 + s * PIVOT_BACKOFF_MM * d      # pivot point along the shaft
            AEdeg[k] = _angle_between(d, Bb - W)  # Angle Error...

        TLE_surface = TLE_center - bead_r

        group_label = np.full(n, labels[b], dtype=object)
        for i in range(n):
            results_rows.append({
                "User": sheet_name,
                "No": int(nos[i]),
                "group": group_label[i],
                "TLE_center": TLE_center[i],
                "TLE_surface": TLE_surface[i],
                "TLEx": TLEx[i],
                "TLEy": TLEy[i],
                "AE_deg": AEdeg[i],
            })

        TLE_group[b]["no"] = TLE_surface
        TLE_group[b]["ae"] = AEdeg

    T = pd.DataFrame(results_rows, columns=["User","No","group","TLE_center","TLE_surface","TLEx","TLEy","AE_deg"])

    # ---------------- Figures ----------------
    # 3D combined plot
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111, projection="3d")
    all_pts_for_bounds = []

    for b in range(2):
        color = colors[b]
        B   = all_B[b]; P1s = all_P1[b]; P2s = all_P2[b]; P3s = all_P3[b]; nos = all_No[b]
        for k in range(B.shape[0]):
            # points
            ax3.scatter(P1s[k,0], P1s[k,1], P1s[k,2], marker='s', s=30, c=[color], edgecolors='k')
            ax3.scatter(P2s[k,0], P2s[k,1], P2s[k,2], marker='^', s=30, c=[color], edgecolors='k')
            ax3.scatter(P3s[k,0], P3s[k,1], P3s[k,2], marker='v', s=30, c=[color], edgecolors='k')
            ax3.scatter(B[k,0],  B[k,1],  B[k,2],  marker='o', s=30, c='w', edgecolors=[color])

            d = best_fit_direction(np.vstack([P1s[k,:], P2s[k,:], P3s[k,:]]))
            t_proj = np.array([
                0.0,
                np.dot(P2s[k,:]-P1s[k,:], d),
                np.dot(P3s[k,:]-P1s[k,:], d),
                np.dot(B[k,:]-P1s[k,:],  d),
            ])
            t = np.linspace(t_proj.min()-10, t_proj.max()+10, 80)
            L = P1s[k,:] + np.outer(t, d)
            ax3.plot(L[:,0], L[:,1], L[:,2], '-', linewidth=1.2, color=color)

            ax3.text(B[k,0]+0.5, B[k,1]+0.5, B[k,2]+0.5,
                     f'{labels[b]}-{int(nos[k])}', color=color, fontsize=8, fontweight="bold")

            all_pts_for_bounds.append(P1s[k,:])
            all_pts_for_bounds.append(P2s[k,:])
            all_pts_for_bounds.append(P3s[k,:])
            all_pts_for_bounds.append(B[k,:])

    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
    ax3.set_title("Pin axes and beads")
    set_axes_equal_3d(ax3, np.vstack(all_pts_for_bounds))
    # make legend handles
    ax3.plot([], [], [], '-', color=C_NO,  linewidth=1.5, label="No Navigation")
    ax3.plot([], [], [], '-', color=C_NAV, linewidth=1.5, label="With Navigation")
    ax3.legend(loc="best")
    fig3.tight_layout()
    fig3.savefig(os.path.join(outdir, f'pin_axes_3d_{safename(sheet_name)}.png'), dpi=200)
    plt.close(fig3)

    # Pin-by-pin 3D panels
    panels = []
    for b in range(2):
        P1s = all_P1[b]; P2s = all_P2[b]; P3s = all_P3[b]; B = all_B[b]; nos = all_No[b]
        for k in range(B.shape[0]):
            d = best_fit_direction(np.vstack([P1s[k,:], P2s[k,:], P3s[k,:]]))
            short = "No" if b == 0 else "Nav"   # compact group tag
            clr   = colors[b]                    # C_NO or C_NAV
            panels.append((P1s[k,:], P2s[k,:], P3s[k,:], B[k,:], d, int(nos[k]), short, clr))

    N = len(panels)
    ncols = min(7, N) if N > 0 else 1
    nrows = int(np.ceil(N / ncols))
    figp = plt.figure(figsize=(1.8*ncols, 1.8*nrows))

    for i, (P1, P2, P3, Bp, d, no, short, clr) in enumerate(panels, start=1):
        ax = figp.add_subplot(nrows, ncols, i, projection="3d")

        # points (color by group)
        ax.scatter([P1[0], P2[0], P3[0], Bp[0]],
                [P1[1], P2[1], P3[1], Bp[1]],
                [P1[2], P2[2], P3[2], Bp[2]],
                marker='o', s=16, facecolors='none', edgecolors=[clr], linewidths=1.0)

        # shaft line (color by group)
        t_proj = np.array([0.0, np.dot(P2-P1, d), np.dot(P3-P1, d), np.dot(Bp-P1, d)])
        t = np.linspace(t_proj.min()-10, t_proj.max()+10, 60)
        L = P1 + np.outer(t, d)
        ax.plot(L[:,0], L[:,1], L[:,2], '-', linewidth=1.2, color=clr)

        # compact in-plot tag near the bead
        ax.text(Bp[0], Bp[1], Bp[2], f'{short}-{no}', color=clr, fontsize=7, fontweight='bold')

        # panel title shows group + row number
        ax.set_title(f'{short} · Pin {no}', fontsize=8)

        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        set_axes_equal_3d(ax, np.vstack([P1, P2, P3, Bp, L]))

    # small legend and title
    legend_handles = [
        plt.Line2D([0],[0], color=C_NO,  lw=2, label='NoNav'),
        plt.Line2D([0],[0], color=C_NAV, lw=2, label='Nav'),
    ]
    figp.legend(handles=legend_handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0))
    figp.suptitle(f'Pin Axes Panels — {sheet_name}', fontsize=10)

    # figp.tight_layout(rect=[0, 0.06, 1, 0.98])  # room for legend & title
    # figp.savefig(os.path.join(outdir, f'pin_axes_panels_{safename(sheet_name)}.png'), dpi=200)
    
    figp.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.16)
    figp.savefig(os.path.join(outdir, f'pin_axes_panels_{safename(sheet_name)}.png'),
             dpi=200, bbox_inches='tight')
    plt.close(figp)


    # Grouped bar plot with AE stars
    TLE_no = TLE_group[0]["no"];  AE_no = TLE_group[0]["ae"]
    TLE_nv = TLE_group[1]["no"];  AE_nv = TLE_group[1]["ae"]
    n_no = len(TLE_no); n_nv = len(TLE_nv)

    bars_vec = list(TLE_no) + [0, 0] + list(TLE_nv)
    x = np.arange(1, len(bars_vec) + 1)

    fig, axL = plt.subplots(figsize=(12, 6))
    rects = axL.bar(x, bars_vec)
    # color groups
    for i in range(n_no):
        rects[i].set_color(C_NO)
    for j in range(n_nv):
        rects[n_no + 2 + j].set_color(C_NAV)

    axL.set_title(STUDY_NAME, fontsize=16)
    axL.set_ylabel("Target Localization Error (mm)", fontsize=14)
    axL.set_ylim(0, YMAX_LEFT)
    axL.set_xticks(x)
    xticklabels = [str(i) for i in range(1, n_no+1)] + ["", ""] + [str(i) for i in range(1, n_nv+1)]
    axL.set_xticklabels(xticklabels, fontsize=12)
    # Guides
    axL.axhline(GUIDE1, linestyle="--", linewidth=2, color=C_NO, label="3 mm guide")
    axL.axhline(GUIDE2, linestyle="--", linewidth=2, color="r",   label="5 mm guide")

    # Value labels centered over bars using rect endpoints (robust)
    for i in range(len(rects)):
        height = rects[i].get_height()
        if height > 0:
            axL.text(rects[i].get_x() + rects[i].get_width()/2., height,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=12)

    # Summary boxes
    avg_no, std_no = mean_safe(TLE_no), std_safe(TLE_no)
    avg_nv, std_nv = mean_safe(TLE_nv), std_safe(TLE_nv)
    axL.text(0.20, 0.90, f'No Navigation\nTLE = {avg_no:.2f}±{std_no:.2f} mm',
             transform=axL.transAxes, fontsize=14, va='top')
    axL.text(0.62, 0.90, f'With Navigation\nTLE = {avg_nv:.2f}±{std_nv:.2f} mm',
             transform=axL.transAxes, fontsize=14, va='top')

    # Right axis AE stars
    axR = axL.twinx()
    axR.set_ylim(0, YMAX_RIGHT)
    axR.set_ylabel("Angular Error (deg)", fontsize=14, color="k")
    axR.tick_params(axis='y', labelcolor='k')

    # AE: indices of real bars
    x_no = x[:n_no]
    x_nv = x[n_no+2:n_no+2+n_nv]
    axR.plot(x_no, AE_no,  '*', markersize=10, color=C_NO,  label="AE (No Nav)")
    axR.plot(x_nv, AE_nv,  '*', markersize=10, color=C_NAV, label="AE (With Nav)")

    # Legend
    handles = [
        plt.Line2D([0],[0], marker='*', color='w', markerfacecolor=C_NO,  markersize=10, label='AE (No Nav)'),
        plt.Line2D([0],[0], marker='*', color='w', markerfacecolor=C_NAV, markersize=10, label='AE (With Nav)'),
        plt.Line2D([0],[0], linestyle='--', color=C_NO, linewidth=2, label='3 mm guide'),
        plt.Line2D([0],[0], linestyle='--', color='r',   linewidth=2, label='5 mm guide'),
    ]
    axL.legend(handles=handles, loc="upper left")

    # p-values
    p_tle = ranksum_safe(TLE_no, TLE_nv)
    p_ang = ranksum_safe(AE_no,  AE_nv)
    # fig.text(0.5, 0.04, f'User: {sheet_name}    p(TLE)={p_tle:.4f}    p(Angle)={p_ang:.4f}',
    #          ha='center', fontsize=11)
    # fig.tight_layout(rect=[0, 0.05, 1, 1])
    # fig.savefig(os.path.join(outdir, f'TLE_bars_two_trials_{safename(sheet_name)}.png'), dpi=200)
    
    axL.text(0.5, -0.22, f'User: {sheet_name}    p(TLE)={p_tle:.4f}    p(Angle)={p_ang:.4f}',
         transform=axL.transAxes, ha='center', fontsize=11)
    fig.subplots_adjust(left=0.10, right=0.96, top=0.90, bottom=0.28)
    fig.savefig(os.path.join(outdir, f'TLE_bars_two_trials_{safename(sheet_name)}.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Save per-sheet CSV
    out_csv = os.path.join(outdir, f'results_table_{safename(sheet_name)}.csv')
    T.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote {out_csv}")

    # stats dict
    stats = dict(user=sheet_name,
                 n_no=int(n_no), n_nav=int(n_nv),
                 avg_no=avg_no, std_no=std_no,
                 avg_nav=avg_nv, std_nav=std_nv,
                 p_tle=p_tle, p_angle=p_ang)
    return T, stats

# -------------------- Driver for all sheets --------------------

def run_all_users(workbook: str):
    if not os.path.exists(workbook):
        raise FileNotFoundError(workbook)

    os.makedirs(ROOT_OUTDIR, exist_ok=True)

    # Collect sheet names
    xl = pd.ExcelFile(workbook)
    sheets = list(xl.sheet_names)

    all_rows = []
    summaries = []

    for s in sheets:
        try:
            user_out = os.path.join(ROOT_OUTDIR, safename(s))
            T, stats = analyze_sheet(s, workbook, user_out)
            all_rows.append(T)
            summaries.append(pd.DataFrame([stats]))
            print(f"[OK] {s}")
        except Exception as e:
            warnings.warn(f"[SKIP] {s}: {e}")

    if all_rows:
        Tall = pd.concat(all_rows, ignore_index=True)
        Tall_path = os.path.join(ROOT_OUTDIR, "results_all_users.csv")
        Tall.to_csv(Tall_path, index=False)
    else:
        Tall = pd.DataFrame()
        Tall_path = os.path.join(ROOT_OUTDIR, "results_all_users.csv")
        Tall.to_csv(Tall_path, index=False)

    if summaries:
        S = pd.concat(summaries, ignore_index=True)
        S_path = os.path.join(ROOT_OUTDIR, "summary_by_user.csv")
        S.to_csv(S_path, index=False)
    else:
        S = pd.DataFrame()
        S_path = os.path.join(ROOT_OUTDIR, "summary_by_user.csv")
        S.to_csv(S_path, index=False)

    print("[DONE] Wrote:\n  {}\n  {}".format(Tall_path, S_path))
    return Tall, S

#================================== Main() =================================================================
def main():
    parser = argparse.ArgumentParser(description="Run TLE analysis for all sheets.")
    parser.add_argument("--workbook", "-w", type=str, default=DEFAULT_WORKBOOK,
                        help="Path to Excel workbook (default: 'TLE Data.xlsx').")
    args = parser.parse_args()
    run_all_users(args.workbook)
    

if __name__ == "__main__":
    main()
###############################################################EN2025#################################################

######################################################################################################################






