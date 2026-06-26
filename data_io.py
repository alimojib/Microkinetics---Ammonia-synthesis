"""
data_io.py
==========
Functions for reading all input data from the Excel source file.

Three sheets are read:
    1. "Thermodynamic Properties"   → surface species H(0K), S(0K), Cp(T)
    2. "NASA Polynomials-0% Strain" → 7-coefficient NASA polynomials for
                                      the three gas-phase species NH3, N2, H2
    3. "Kinetic Parameters"         → BEP (alpha, E0) and pre-exponential
                                      factors (alpha_1) for surface reactions

Nothing in this module does any calculation — it only loads raw numbers and
returns them in plain Python dicts for the rest of the project to consume.

Public API
----------
    read_surface_data(excel_path)   -> dict
    read_nasa_data(excel_path)      -> dict
    read_kinetic_params(excel_path) -> dict
"""

import csv as _csv_mod

import pandas as pd
import numpy as np


# ==============================================================================
# SURFACE SPECIES DATA
# ==============================================================================

def read_surface_data(excel_path):
    """
    Read the 'Thermodynamic Properties' sheet and return a dict of
    DFT-derived thermodynamic data for all surface species.

    Sheet layout (zero-indexed rows):
        Row 0  : main header — ignored
        Row 1  : column labels — ignored (column positions are fixed)
        Row 2+ : one surface species per row

    Columns within each data row:
        0      : species name  (string)
        1      : H at 0 K      (kcal/mol)
        2      : S at 0 K      (cal/mol/K)
        3–17   : Cp at the 15 tabulated temperatures T_CP (cal/mol/K)

    Parameters
    ----------
    excel_path : str — path to the Excel file

    Returns
    -------
    dict
        Keys   : species name (str)
        Values : dict with keys
                    "H0" (float, kcal/mol)
                    "S0" (float, cal/mol/K)
                    "Cp" (np.ndarray of shape (15,), cal/mol/K)
    """
    raw = pd.read_excel(
        excel_path,
        sheet_name="Thermodynamic Properties",
        header=None,
    )

    surf_data = {}

    # Iterate over data rows; skip the two header rows (indices 0 and 1)
    for _, row in raw.iloc[2:].iterrows():
        name = str(row.iloc[0]).strip()
        H0   = float(row.iloc[1])                    # kcal/mol at 0 K
        S0   = float(row.iloc[2])                    # cal/(mol·K) at 0 K
        cp   = row.iloc[3:18].astype(float).values   # 15 Cp values

        surf_data[name] = {"H0": H0, "S0": S0, "Cp": cp}

    return surf_data


# ==============================================================================
# NASA POLYNOMIAL DATA (GAS PHASE)
# ==============================================================================

def _read_nasa_row(df, row_idx):
    """
    Extract the seven NASA polynomial coefficients [a1 … a7] from one
    row of the NASA Polynomials sheet.

    The coefficients occupy columns 2–8 (zero-indexed).

    Parameters
    ----------
    df      : pd.DataFrame — the full sheet loaded with header=None
    row_idx : int          — zero-based row index

    Returns
    -------
    np.ndarray of shape (7,)  — [a1, a2, a3, a4, a5, a6, a7]
    """
    return df.iloc[row_idx, 2:9].astype(float).values


def read_nasa_data(excel_path):
    """
    Read the 'NASA Polynomials-0% Strain' sheet and return piecewise
    NASA polynomial data for the three gas-phase species: NH3, N2, H2.

    Each entry holds two sets of coefficients (low / high temperature
    range) and the breakpoint temperature at which the switch occurs.

    Row layout in the sheet (zero-indexed):
        3 : NH3 low range   (298–592.4 K)
        4 : NH3 high range  (592.4–1500 K)
        5 : N2  low range   (298–791.5 K)
        6 : N2  high range  (791.5–1500 K)
        7 : H2  low range   (298–791.5 K)
        8 : H2  high range  (791.5–1500 K)

    Parameters
    ----------
    excel_path : str — path to the Excel file

    Returns
    -------
    dict
        Keys   : species name (str) — "NH3", "N2", or "H2"
        Values : dict with keys
                    "low"     (np.ndarray shape (7,)) — low-T coefficients
                    "high"    (np.ndarray shape (7,)) — high-T coefficients
                    "T_break" (float, K)              — range switchover temp
    """
    raw_nasa = pd.read_excel(
        excel_path,
        sheet_name="NASA Polynomials-0% Strain",
        header=None,
    )

    nasa_gas = {
        "NH3": {
            "low":     _read_nasa_row(raw_nasa, 3),
            "high":    _read_nasa_row(raw_nasa, 4),
            "T_break": 592.4,
        },
        "N2": {
            "low":     _read_nasa_row(raw_nasa, 5),
            "high":    _read_nasa_row(raw_nasa, 6),
            "T_break": 791.5,
        },
        "H2": {
            "low":     _read_nasa_row(raw_nasa, 7),
            "high":    _read_nasa_row(raw_nasa, 8),
            "T_break": 791.5,
        },
    }

    return nasa_gas


# ==============================================================================
# KINETIC PARAMETERS (BEP + PRE-EXPONENTIAL)
# ==============================================================================

# Internal mapping from the Excel reaction label (using " D " as arrow
# shorthand) to the canonical step label used in kinetics.py STEPS list.
# This is needed because the Excel sheet uses a different notation than
# the STEPS definitions in kinetics.py.
_EXCEL_LABEL_TO_STEP_LABEL = {
    "N2(T) + *(T) D 2N(T)"          : "N2(T)+*(T)<=>2N(T)",
    "N(T) + H(T) D NH(T) + *(T)"    : "N(T)+H(T)<=>NH(T)+*(T)",
    "NH(T) + H(T) D NH2(T) + *(T)"  : "NH(T)+H(T)<=>NH2(T)+*(T)",
    "NH2(T) + H(T) D NH3(T) + *(T)" : "NH2(T)+H(T)<=>NH3(T)+*(T)",
    "N2(S) + *(S) D 2N(S)"          : "N2(S)+*(S)<=>2N(S)",
    "N(S) + H(S) D NH(S) + *(S)"    : "N(S)+H(S)<=>NH(S)+*(S)",
    "NH(S) + H(S) D NH2(S) + *(S)"  : "NH(S)+H(S)<=>NH2(S)+*(S)",
    "NH2(S) + H(S) D NH3(S) + *(S)" : "NH2(S)+H(S)<=>NH3(S)+*(S)",
    "N2(S) + *(SL) D N(S) + N(SL)"  : "N2(S)+*(SL)<=>N(S)+N(SL)",
}


def read_nasa_surface_data(excel_path):
    """
    Read the 'NASA Polynomials-0% Strain' sheet and return piecewise NASA-7
    coefficients for all 13 surface species.

    All surface species use T_break = 500 K (low: 100-500 K, high: 500-1500 K).
    Transition-state rows are skipped.

    Row layout in the sheet (zero-indexed):
         9 : N2(T)  low     10 : N2(T)  high
        11 : N(T)   low     12 : N(T)   high
        13 : H(T)   low     14 : H(T)   high
        15 : NH3(T) low     16 : NH3(T) high
        17 : NH2(T) low     18 : NH2(T) high
        19 : NH(T)  low     20 : NH(T)  high
        (rows 21-28: transition states — skipped)
        29 : N2(S)  low     30 : N2(S)  high
        31 : N(S)   low     32 : N(S)   high
        33 : N(SL)  low     34 : N(SL)  high
        35 : H(S)   low     36 : H(S)   high
        37 : NH3(S) low     38 : NH3(S) high
        39 : NH2(S) low     40 : NH2(S) high
        41 : NH(S)  low     42 : NH(S)  high

    Parameters
    ----------
    excel_path : str — path to the Excel file

    Returns
    -------
    dict
        Keys   : species name (str)
        Values : dict with keys
                    "low"     (np.ndarray shape (7,)) — low-T coefficients
                    "high"    (np.ndarray shape (7,)) — high-T coefficients
                    "T_break" (float, K = 500.0)
    """
    raw = pd.read_excel(
        excel_path,
        sheet_name="NASA Polynomials-0% Strain",
        header=None,
    )

    _SURF_ROWS = {
        "N2(T)":  ( 9, 10),
        "N(T)":   (11, 12),
        "H(T)":   (13, 14),
        "NH3(T)": (15, 16),
        "NH2(T)": (17, 18),
        "NH(T)":  (19, 20),
        "N2(S)":  (29, 30),
        "N(S)":   (31, 32),
        "N(SL)":  (33, 34),
        "H(S)":   (35, 36),
        "NH3(S)": (37, 38),
        "NH2(S)": (39, 40),
        "NH(S)":  (41, 42),
    }

    nasa_surf = {}
    for name, (row_low, row_high) in _SURF_ROWS.items():
        nasa_surf[name] = {
            "low":     _read_nasa_row(raw, row_low),
            "high":    _read_nasa_row(raw, row_high),
            "T_break": 500.0,
        }

    return nasa_surf


def read_nasa_from_csv(csv_path):
    """
    Read NASA-7 polynomial coefficients from data_ammonia.csv and return
    (nasa_gas, nasa_surf) in the same dict format as read_nasa_data() and
    read_nasa_surface_data().

    The CSV stores raw coefficients that SeqSim adjusts at runtime via
    Strain_Coef_H and Strain_Coef_S (LSR + strain corrections).  At zero
    strain and Q_target = Q_ref (pure Ru reference), the corrections reduce to
    a constant offset on a6 and a7 for each surface species.  This function
    applies those offsets once so the returned dicts are ready for direct use
    in nasa_H / nasa_S — no further correction needed at call time.

    Parameters
    ----------
    csv_path : str — path to data_ammonia.csv

    Returns
    -------
    nasa_gas  : dict  — same structure as read_nasa_data()
    nasa_surf : dict  — same structure as read_nasa_surface_data()
    """
    # ── Parse CSV into sections ────────────────────────────────────────────────
    data = {}
    with open(csv_path, mode='r') as f:
        reader = _csv_mod.reader(f)
        section = None
        rows = []
        for row in reader:
            row = [v.strip() for v in row]
            if not any(row):
                continue
            if len(row) == 1:
                if section and rows:
                    arr = np.array(rows, dtype=float)
                    data[section] = arr.flatten() if arr.shape[0] == 1 else arr
                    rows = []
                section = row[0]
            elif section:
                try:
                    rows.append([float(v) for v in row])
                except ValueError:
                    pass
        if section and rows:
            arr = np.array(rows, dtype=float)
            data[section] = arr.flatten() if arr.shape[0] == 1 else arr

    def _c(key):
        """Return a fresh copy of the 7-element coefficient array for key."""
        return data[key].flatten()[:7].copy()

    # ── Gas phase: no corrections (gas species are unaffected by LSR/strain) ──
    nasa_gas = {
        "N2":  {"low": _c("A_N2_l"),  "high": _c("A_N2_h"),  "T_break": 791.5},
        "H2":  {"low": _c("A_H2_l"),  "high": _c("A_H2_h"),  "T_break": 791.5},
        "NH3": {"low": _c("A_NH3_l"), "high": _c("A_NH3_h"), "T_break": 592.4},
    }

    # ── Surface species: apply zero-strain corrections to a6 (idx 5) and a7 (idx 6)
    #
    # At strain=0 and A6_LSR=0 (Q_target=Q_ref), SeqSim's amm_thermo4 does:
    #   A6_Correction = A6_LSR - A6_Strain = 0 - A6_Strain  →  a6 += A6_Strain
    #   a7 += A7_Strain
    # where A6_Strain = Strain_Coef_H @ [0, 1]  (second column of the matrix)
    #       A7_Strain = Strain_Coef_S @ [0, 0, 1]  (third column)
    #
    # Correction index → surface species (SeqSim ordering, 13 species):
    #   0-5 : N2(T), N(T), H(T), NH3(T), NH2(T), NH(T)
    #   6-11: N2(S), N(S), H(S),  NH3(S), NH2(S), NH(S)
    #   12  : N(SL)
    A6_corr = (data['Strain_Coef_H'] @ np.array([0.0, 1.0]))        # (13,)
    A7_corr = (data['Strain_Coef_S'] @ np.array([0.0, 0.0, 1.0]))   # (13,)

    def _surf(key_l, key_h, ci):
        """Build one corrected surface-species entry."""
        low  = _c(key_l);  low[5]  += A6_corr[ci];  low[6]  += A7_corr[ci]
        high = _c(key_h);  high[5] += A6_corr[ci];  high[6] += A7_corr[ci]
        return {"low": low, "high": high, "T_break": 500.0}

    nasa_surf = {
        "N2(T)":  _surf("A_N2s1_l",  "A_N2s1_h",   0),
        "N(T)":   _surf("A_Ns1_l",   "A_Ns1_h",    1),
        "H(T)":   _surf("A_Hs1_l",   "A_Hs1_h",    2),
        "NH3(T)": _surf("A_NH3s1_l", "A_NH3s1_h",  3),
        "NH2(T)": _surf("A_NH2s1_l", "A_NH2s1_h",  4),
        "NH(T)":  _surf("A_NHs1_l",  "A_NHs1_h",   5),
        "N2(S)":  _surf("A_N2s2_l",  "A_N2s2_h",   6),
        "N(S)":   _surf("A_Ns2_l",   "A_Ns2_h",    7),
        "H(S)":   _surf("A_Hs2_l",   "A_Hs2_h",    8),
        "NH3(S)": _surf("A_NH3s2_l", "A_NH3s2_h",  9),
        "NH2(S)": _surf("A_NH2s2_l", "A_NH2s2_h", 10),
        "NH(S)":  _surf("A_NHs2_l",  "A_NHs2_h",  11),
        "N(SL)":  _surf("A_Ns3_l",   "A_Ns3_h",   12),
    }

    return nasa_gas, nasa_surf


def read_kinetic_params(excel_path):
    """
    Read the 'Kinetic Parameters' sheet and return the BEP parameters
    (alpha, E0) and pre-exponential factor (alpha_1) for each surface
    reaction step that has kinetic data in the Excel file.

    Sheet layout (zero-indexed rows and columns):

        Table 6 — BEP parameters:
            Rows 4–12, Column 2  : reaction label (Excel notation)
            Rows 4–12, Column 3  : alpha  (BEP slope, dimensionless)
            Rows 4–12, Column 4  : E0     (BEP intercept, kcal/mol)

        Table 8 — Pre-exponential scaling:
            Rows 20–28, Column 2 : reaction label (same Excel notation)
            Rows 20–28, Column 5 : alpha_1  (pre-exponential A, 1/s)

    The Excel labels use " D " as the arrow (shorthand for ⇌).
    These are translated to the canonical STEPS labels on the way out.

    Parameters
    ----------
    excel_path : str — path to the Excel file

    Returns
    -------
    dict
        Keys   : canonical step label (str), matching kinetics.STEPS labels
        Values : dict with keys
                    "alpha"  (float)  — BEP slope [-]
                    "E0"     (float)  — BEP intercept [kcal/mol]
                    "A"      (float)  — pre-exponential factor [1/s]
    """
    raw = pd.read_excel(
        excel_path,
        sheet_name="Kinetic Parameters",
        header=None,
    )

    kin_params = {}

    # ── Table 6: read BEP alpha and E0 (rows 4–12) ───────────────────────────
    #
    # Row layout within Table 6 (zero-indexed):
    #   col 2 : reaction label string
    #   col 3 : alpha (BEP slope)
    #   col 4 : E0    (BEP intercept, kcal/mol)
    #
    # The 9 rows (4–12) correspond to the 9 surface reactions that have
    # BEP parameters: N2 dissociation on terrace and step, and all 6
    # hydrogenation steps plus the step N2 dissociation to SL site.

    BEP_ROW_START = 4
    BEP_ROW_END   = 13   # exclusive — range(4, 13) gives rows 4..12

    for row_idx in range(BEP_ROW_START, BEP_ROW_END):
        excel_label = str(raw.iloc[row_idx, 2]).strip()
        alpha       = float(raw.iloc[row_idx, 3])
        E0          = float(raw.iloc[row_idx, 4])

        # Translate Excel label → canonical STEPS label
        canonical_label = _EXCEL_LABEL_TO_STEP_LABEL.get(excel_label)

        if canonical_label is None:
            # Label not in our mapping — skip gracefully rather than crash.
            # This should not happen if the mapping dict above is complete.
            continue

        # Initialise entry; alpha_1 will be filled in from Table 8 below
        kin_params[canonical_label] = {
            "alpha" : alpha,
            "E0"    : E0,
            "A"     : None,   # placeholder; filled by Table 8 loop below
        }

    # ── Table 8: read pre-exponential alpha_1 (rows 20–28) ───────────────────
    #
    # Row layout within Table 8 (zero-indexed):
    #   col 2 : reaction label string (same Excel notation as Table 6)
    #   col 5 : alpha_1  (the pre-exponential factor A, units 1/s)
    #
    # The row order matches Table 6 exactly (same 9 reactions).

    PREEXP_ROW_START = 20
    PREEXP_ROW_END   = 29   # exclusive — range(20, 29) gives rows 20..28

    for row_idx in range(PREEXP_ROW_START, PREEXP_ROW_END):
        excel_label = str(raw.iloc[row_idx, 2]).strip()
        alpha_1     = float(raw.iloc[row_idx, 5])

        canonical_label = _EXCEL_LABEL_TO_STEP_LABEL.get(excel_label)

        if canonical_label is None:
            continue

        if canonical_label in kin_params:
            # Store alpha_1 as the pre-exponential A
            kin_params[canonical_label]["A"] = alpha_1

    return kin_params
