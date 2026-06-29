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

# Path to the SeqSim thermodynamic CSV dataset.
# Used when USE_CSV_THERMO = True (see below).
CSV_PATH = "data_ammonia.csv"

# When True, load NASA-7 gas and surface coefficients from data_ammonia.csv
# (SeqSim dataset) instead of from the Excel file.  The zero-strain LSR
# corrections embedded in the CSV are applied automatically so the resulting
# coefficients match what SeqSim uses at strain=0.
# Set to False to revert to the Excel (sciadv) dataset.
USE_CSV_THERMO = True

# Output paths for the three equilibrium figures
FIG_CP_PATH      = "cp_fits.png"
FIG_KEQSTEP_PATH = "keq_elementary.png"
FIG_KEQOV_PATH   = "keq_overall.png"

# Output paths for the microkinetics figures
FIG_MICROKIN_PATH     = "microkinetics.png"
FIG_TOF_PATH          = "tof.png"
FIG_COVERAGE_PATH     = "surface_coverage.png"
FIG_RATES_PATH        = "reaction_rates.png"
FIG_SS_TIME_PATH      = "ss_time.png"


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
T_ARR = np.arange(573, 823 + 1, 5, dtype=float)

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
# MICROKINETICS — PRE-EXPONENTIAL FACTOR OVERRIDE
# ==============================================================================

# Set USE_UNIFORM_A = True  to use a single A for all surface reactions
#     (matching Code 3 / SequentialSimulation.py, useful for direct comparison).
# Set USE_UNIFORM_A = False to use the reaction-specific A values read from the
#     Excel file (the physically derived values — default).
#
# The Excel-derived A values are always loaded; this flag only controls whether
# they are applied or replaced at runtime.
USE_UNIFORM_A = False

# Pre-exponential factor used when USE_UNIFORM_A = True [s⁻¹].
# Value taken from Code 3 (SequentialSimulation.py, Params.A = 1.56e19).
A_UNIFORM = 1.56e19


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

# Activation energies for inter-site diffusion steps [kcal/mol].
# Source: SequentialSimulation.py (SeqSim), at zero strain.
# Formula in SeqSim: Ea = strain_coeff * (strain*100) + E0; values below are E0.
EA_N_DIFF   = 20.20056   # steps 6 & 15 — N(T)<=>N(S)+*(T) / N(T)<=>N(SL)+*(T)
EA_NH_DIFF  = 15.03512   # step  7      — NH(T)+*(S)<=>NH(S)+*(T)
EA_NH2_DIFF =  5.34992   # step  8      — NH2(T)+*(S)<=>NH2(S)+*(T)
EA_NH3_DIFF = 13.18263   # step  9      — NH3(T)+*(S)<=>NH3(S)+*(T)

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
# Capped at 1e3 s to prevent the Radau solver from stepping over fast
# transients when the surface is in a N-poisoned state at high temperature.
ODE_MAX_STEP = 1e3

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
ODE_ATOL_SURF = 1.0e-27   # mol/cm²  — for all 13 surface coverage variables
# Was 1e-30; relaxed ×1000 after surface kf ×ABYV change so atol stays well
# below SS_TOL and doesn't falsely stall the solver on tiny residuals.
ODE_ATOL_GAS  = 1.0e-18   # mol/cm³  — for all 3 gas-phase concentration variables

# Minimum allowed vacant-site concentration [mol/cm²].
# The site balance can drift slightly negative due to ODE solver trial steps.
# A hard max(..., 0) clip introduces a kink in the ODE right-hand side that
# breaks the implicit solver's Jacobian estimate and causes step-size collapse.
# This tiny positive floor eliminates the kink while being physically
# indistinguishable from zero (it is ~10^20 times smaller than SDEN_T).
EPS_VAC = 1.0e-30   # mol/cm²

# Steady-state detection tolerance [mol/(cm²·s) and mol/(cm³·s)].
# Raised to 1e-10 for two compounding reasons:
#   1. Surface kf ×ABYV change made rates ~1200× larger → SS residual dy/dt
#      is proportionally larger than the original 1e-25 calibration.
#   2. H2 dissociative kf = Stick/SDTOT² gives large rf ≈ rb at true SS;
#      machine-arithmetic cancellation leaves a residual ~2×ε_mach×rf of
#      ~1e-15 to 1e-13 mol/cm²·s at high temperature (low H coverage).
#      SS_TOL must exceed this floor or the event never fires even when the
#      system is at genuine physical steady state.
SS_TOL = 1.0e-13   # mol/(cm²·s) or mol/(cm³·s)


# ==============================================================================
# COLD-START DIAGNOSTIC
# ==============================================================================

# When RUN_COLD_START = True, an extra ODE integration is run at COLD_START_T
# using a fresh equilibrated surface (no warm-start inheritance from lower T).
# The result is printed side-by-side with the warm-start value at the same T,
# which reveals whether bistability or warm-start history causes differences
# between main and SeqSim (SeqSim always starts cold).
RUN_COLD_START = True
COLD_START_T   = 593.0   # K — temperature for the cold-start diagnostic run


# ==============================================================================
# CSTR REACTOR OPERATING CONDITIONS
# ==============================================================================

# Individual species volumetric flow rates at standard conditions [cm³/s].
# Matched to SeqSim: V = 1 cm³, Q_total = 1 cm³/s → τ = V/Q = 1 s.
# Feed composition kept at 3:1 H2:N2 (x_N2 = 0.25, x_H2 = 0.75).
Q_IN  = 1.0    # cm³/s  total inlet flow  (SeqSim: Q = 1 cm³/s)
Q_OUT = Q_IN   # cm³/s  outlet = inlet (isobaric)

# Reactor volume [cm³]
V_REACTOR = 1.0   # cm³  (SeqSim: V = 1 cm³  →  τ = 1 s)

# Feed mole fractions: 3:1 H2:N2, no NH3 — matches SeqSim feed composition.
X_N2_FEED  = 0.25   # N2 mole fraction
X_H2_FEED  = 0.75   # H2 mole fraction
X_NH3_FEED = 0.0    # no NH3 in feed
