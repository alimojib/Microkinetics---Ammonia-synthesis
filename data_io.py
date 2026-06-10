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
