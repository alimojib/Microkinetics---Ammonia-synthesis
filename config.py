"""
config.py
=========
Central configuration for the NH3 equilibrium + microkinetics project.

All physical constants, file paths, and numerical settings live here.
Every other module imports from this file, so changing a value once
propagates everywhere automatically.

Usage
-----
    from config import R_CAL, T_ARR, EXCEL_PATH
"""

import numpy as np

# ==============================================================================
# FILE PATHS
# ==============================================================================

# Path to the source Excel file.
# Update this string to point to the file on your machine.
EXCEL_PATH = r"C:\Users\AliMojibpour\Downloads\sciadv.abl6576_dataset_file.xlsx"

# Output paths for the three equilibrium figures
FIG_CP_PATH      = "cp_fits.png"
FIG_KEQSTEP_PATH = "keq_elementary.png"
FIG_KEQOV_PATH   = "keq_overall.png"

# Output paths for the microkinetics figures
FIG_MICROKIN_PATH     = "microkinetics.png"
FIG_TOF_PATH          = "tof.png"
FIG_COVERAGE_PATH     = "surface_coverage.png"
FIG_RATES_PATH        = "reaction_rates.png"


# ==============================================================================
# PHYSICAL CONSTANTS
# ==============================================================================

# Universal gas constant in calories — used for ΔG and Keq [cal/(mol·K)]
R_CAL = 1.987

# Universal gas constant in SI units — used in Hertz-Knudsen [J/(mol·K)]
# and for converting pressure to concentration via ideal gas law.
R_SI = 8.314   # J/(mol·K)  ≡  Pa·m³/(mol·K)

# Universal gas constant in cm³·bar/(mol·K) — used for ideal-gas C = P/(R*T)
# Derived:  R_SI [J/(mol·K)] × 10 [cm³·bar / J]  =  83.14 cm³·bar/(mol·K)
R_CM3_BAR = 83.14   # cm³·bar/(mol·K)

# Reference pressure [bar]
P_BAR = 50.0


# ==============================================================================
# TEMPERATURE SETTINGS
# ==============================================================================

# Temperature array for Keq and microkinetics evaluation [K]
# Covers the industrially relevant 573–873 K window at 5 K (≈ 5°C) increments.
# np.arange is used instead of linspace so the step size is exact.
# +1 on the stop ensures 873 K is included (arange stop is exclusive).
T_ARR = np.arange(573, 873 + 1, 5, dtype=float)

# Tabulated Cp temperature points used in the Excel dataset [K]
# These are the 15 fixed points at which DFT-derived Cp values are reported.
T_CP = np.array(
    [100, 200, 300, 400, 500, 600, 700, 800,
     900, 1000, 1100, 1200, 1300, 1400, 1500],
    dtype=float,
)


# ==============================================================================
# POLYNOMIAL FIT SETTINGS
# ==============================================================================

# Degree of the polynomial fitted to surface-species Cp(T) data.
# Degree 3 was chosen as a good balance between flexibility and over-fitting
# across the 100–1500 K range covered by T_CP.
CP_POLY_DEGREE = 3


# ==============================================================================
# SPECIES THAT REPRESENT VACANT SURFACE SITES
# ==============================================================================

# Vacant sites carry no enthalpy or entropy (reference state = 0).
# Defined here so the same set is used consistently across all modules.
VACANT_SITES = frozenset({"*T", "*S", "*SL"})


# ==============================================================================
# MICROKINETICS — REACTOR AND CATALYST GEOMETRY
# ==============================================================================

# Total surface site density [mol/cm²]
# This is the combined terrace + step site density reported in the paper.
SDTOT = 2.6188e-9   # mol/cm²

# Fraction of total sites that are step sites [-]
# The remaining fraction (1 - RATIO_S) are terrace sites.
RATIO_S = 0.02

# Catalyst surface area per unit reactor volume [cm²_cat / cm³_reac]
ABYV = 1200.0   # cm²/cm³

# Site density for terrace sites [mol/cm²]
# Terrace sites make up the majority of the surface.
SDEN_T = (1.0 - RATIO_S) * SDTOT   # mol/cm²

# Site density for step sites [mol/cm²]
# Step sites are the minority but are the primary N2 dissociation pathway.
SDEN_S = RATIO_S * SDTOT            # mol/cm²


# ==============================================================================
# MICROKINETICS — HERTZ-KNUDSEN ADSORPTION PARAMETERS
# ==============================================================================

# Sticking coefficient [-] used in the Hertz-Knudsen expression.
# Set to 0.5 for all adsorbing species as a baseline assumption.
# This is a dimensionless probability that a gas-phase collision results
# in adsorption.
STICKING_COEFF = 0.5

# Activation energy for adsorption [kcal/mol].
# Assumed to be zero for all physisorption/chemisorption adsorption steps —
# i.e., no barrier to adsorption (the incoming molecule is not activated).
EA_ADS = 0.0   # kcal/mol

# Molecular weights of the three gas-phase species [g/mol].
# Used in the Hertz-Knudsen expression: kf ∝ 1/sqrt(MW).
MW = {
    "N2":  28.014,   # g/mol
    "H2":   2.016,   # g/mol
    "NH3": 17.031,   # g/mol
}


# ==============================================================================
# MICROKINETICS — ODE SOLVER SETTINGS
# ==============================================================================

# Time span for the ODE integration [s].
# Start at t=0, integrate to T_FINAL to reach steady state.
T_SPAN = (0.0, 1.0e6)   # seconds

# Maximum ODE solver step size [s].
# A smaller value increases accuracy but also computation time.
# None lets the solver choose adaptively.
ODE_MAX_STEP = np.inf

# ODE solver method.
# "Radau" is an implicit Runge-Kutta method specifically designed for stiff
# systems.  It is more robust than LSODA on microkinetic problems where the
# stiffness ratio (fastest / slowest timescale) exceeds ~10^10, because LSODA
# can get stuck oscillating between its Adams and BDF sub-methods on such
# problems.  Radau is slower per step but takes far fewer failed steps.
ODE_METHOD = "Radau"

# Relative tolerance for the stiff ODE solver.
ODE_RTOL = 1.0e-8

# Absolute tolerance vector — one value per ODE variable.
# A single scalar atol cannot work well here because surface coverages
# (~1e-12 to 1e-9 mol/cm²) and gas concentrations at 50 bar (~1e-3 mol/cm³)
# differ by ~10 orders of magnitude.  If atol is tight enough for surface
# species it is unnecessarily tight for gas; if loose enough for gas it
# accepts steps that corrupt the surface state entirely.
# The solution is a per-variable vector passed directly to solve_ivp.
# These two constants are combined into the vector inside run_microkinetics().
ODE_ATOL_SURF = 1.0e-30   # mol/cm²  — for all 13 surface coverage variables
ODE_ATOL_GAS  = 1.0e-18   # mol/cm³  — for all 3 gas-phase concentration variables

# Minimum allowed vacant-site concentration [mol/cm²].
# The site balance can drift slightly negative due to ODE solver trial steps.
# A hard max(..., 0) clip introduces a kink in the ODE right-hand side that
# breaks the implicit solver's Jacobian estimate and causes step-size collapse.
# This tiny positive floor eliminates the kink while being physically
# indistinguishable from zero (it is ~10^20 times smaller than SDEN_T).
EPS_VAC = 1.0e-30   # mol/cm²

# Steady-state detection tolerance [mol/(cm²·s) and mol/(cm³·s)].
# Lowered from 1e-20 to 1e-25 because at 50 bar the gas-phase dy/dt is
# ~50× larger than at 1 bar even at genuine steady state, so the previous
# threshold was firing too early (before true steady state was reached).
SS_TOL = 1.0e-25   # mol/(cm²·s) or mol/(cm³·s)


# ==============================================================================
# CSTR REACTOR OPERATING CONDITIONS
# ==============================================================================

# Individual species volumetric flow rates at standard conditions [cm³/s].
# H2 = 34.2 sccm, N2 = 11.4 sccm  →  total = 45.6 sccm = 0.76 cm³/s
# (1 sccm = 1 cm³/min  →  divide by 60 to convert to cm³/s)
Q_H2_SCCM = 34.2                         # cm³/min (sccm)
Q_N2_SCCM = 11.4                         # cm³/min (sccm)
Q_IN      = (Q_H2_SCCM + Q_N2_SCCM) / 60.0   # cm³/s  total inlet flow
Q_OUT     = Q_IN                         # cm³/s  outlet = inlet (isobaric)

# Reactor volume [cm³]
V_REACTOR = 7.7   # cm³

# Feed mole fractions derived from the flow rates above.
# x_N2 = 11.4/45.6 = 0.25, x_H2 = 34.2/45.6 = 0.75, x_NH3 = 0.0
# These are consistent with the 3:1 H2:N2 feed already used in the ICs.
# Stored here for transparency; used in odes_rhs() to compute C_feed.
X_N2_FEED  = Q_N2_SCCM / (Q_N2_SCCM + Q_H2_SCCM)   # 0.25
X_H2_FEED  = Q_H2_SCCM / (Q_N2_SCCM + Q_H2_SCCM)   # 0.75
X_NH3_FEED = 0.0                                      # no NH3 in feed
