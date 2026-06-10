"""
main.py
=======
Entry point for the NH3 equilibrium + microkinetics project.

This file does nothing but call the right functions in the right order.
All logic lives in the five supporting modules:

    config.py         — constants, paths, settings
    data_io.py        — Excel reading (thermo + kinetic parameters)
    thermodynamics.py — Cp fits, H/S functions, step Keq calculation
    kinetics.py       — step definitions, stoichiometry matrix, overall Keq
    microkinetics.py  — rate constants, ODE system, steady-state solver
    plotting.py       — all figures

Run
---
    python main.py

Output
------
    cp_fits.png         — Cp(T) fits for surface and gas-phase species
    keq_elementary.png  — Keq(T) for each of the 22 elementary steps
    keq_overall.png     — DFT overall Keq vs literature Keq
    microkinetics.png   — steady-state coverages and TOF vs temperature
    Console report      — linear-combination multipliers + microkinetics summary
"""

import warnings
import numpy as np

# Suppress benign integration warnings from scipy.integrate.quad
warnings.filterwarnings("ignore")

# ── Project modules ────────────────────────────────────────────────────────────
from config import (
    EXCEL_PATH,
    T_ARR,
    T_CP,
    SDTOT,          # total site density — used for TOF normalisation
    ABYV,           # area/volume — used for TOF normalisation
    R_CM3_BAR,      # gas constant [cm³·bar/(mol·K)] — for C_feed calculation
    P_BAR,          # total pressure [bar]
    X_N2_FEED,      # feed mole fraction of N2
    X_H2_FEED,      # feed mole fraction of H2
    X_NH3_FEED,     # feed mole fraction of NH3
)

from data_io import (
    read_surface_data,
    read_nasa_data,
    read_kinetic_params,
)

from thermodynamics import (
    fit_surface_cp_polynomials,
    compute_step_keq,
)

from kinetics import (
    STEPS,
    ALL_SPECIES,
    N_STEPS,
    build_stoichiometry_matrix,
    solve_linear_combination,
    compute_overall_keq,
    K_lit,
)

from microkinetics import (
    precompute_kinetics,
    build_initial_conditions,
    build_equilibrated_initial_conditions,
    build_warm_start,
    compute_rate_constants,
    compute_net_rates,
    run_microkinetics,
    N_VARS,
    IDX_N2G, IDX_H2G, IDX_NH3G,
    IDX_NT,  IDX_HT,  IDX_NHT,
    IDX_NH2T, IDX_NH3T, IDX_N2T,
    IDX_NS,  IDX_HS,  IDX_NHS,
    IDX_NH2S, IDX_NH3S, IDX_N2S, IDX_NSL,
)

from plotting import (
    plot_cp_fits,
    plot_elementary_keq,
    plot_overall_keq,
    plot_tof,
    plot_surface_coverage,
    plot_reaction_rates,
)


# ==============================================================================
# STEP 1 — LOAD DATA
# ==============================================================================

surf_data  = read_surface_data(EXCEL_PATH)
nasa_gas   = read_nasa_data(EXCEL_PATH)
kin_params = read_kinetic_params(EXCEL_PATH)


# ==============================================================================
# STEP 2 — FIT SURFACE Cp POLYNOMIALS AND PLOT
# ==============================================================================

cp_poly = fit_surface_cp_polynomials(surf_data)

plot_cp_fits(surf_data, cp_poly, nasa_gas, T_CP)


# ==============================================================================
# STEP 3 — COMPUTE Keq FOR EACH ELEMENTARY STEP AND PLOT
# ==============================================================================

step_Keq = compute_step_keq(STEPS, T_ARR, nasa_gas, surf_data, cp_poly)

plot_elementary_keq(STEPS, step_Keq, T_ARR)


# ==============================================================================
# STEP 4 — SOLVE FOR OVERALL Keq
# ==============================================================================

# Build the species-stoichiometry matrix A  (n_species × n_steps)
A = build_stoichiometry_matrix(STEPS, ALL_SPECIES)

# Find the multipliers ν that combine the active steps into N2+3H2 <=> 2NH3
nu_vec, b_check, residual = solve_linear_combination(A, STEPS, ALL_SPECIES)


# ==============================================================================
# STEP 5 — PRINT EQUILIBRIUM REPORT
# ==============================================================================

sp_idx = {sp: i for i, sp in enumerate(ALL_SPECIES)}

print("\n── Linear combination multipliers (νᵢ) ──")
print(f"  \t{'Step':<44}  {'ν':>8}  Status")
print("  " + "-" * 65)

for ii, (step, nu) in enumerate(zip(STEPS, nu_vec)):
    status = "active" if step.active else "excluded (ν forced = 0)"
    print(f"{ii}\t{step.label:<44}  {nu:>+8.4f}  {status}")

print(f"\nResidual max |Aν - b| = {residual:.2e}")

# Confirm the combined reaction matches N2 + 3H2 → 2NH3 exactly
print("\nNet stoichiometry of the combined reaction:")
for sp in ALL_SPECIES:
    val = b_check[sp_idx[sp]]
    if abs(val) > 1e-6:
        print(f"  {sp}: {val:+.4f}")


# ==============================================================================
# STEP 6 — COMPUTE AND PLOT OVERALL Keq VS LITERATURE
# ==============================================================================

Keq_overall = compute_overall_keq(STEPS, step_Keq, nu_vec)
Keq_lit     = np.array([K_lit(T) for T in T_ARR])

plot_overall_keq(T_ARR, Keq_overall, Keq_lit)


# ==============================================================================
# STEP 7 — PRECOMPUTE RATE CONSTANTS OVER THE FULL TEMPERATURE ARRAY
# ==============================================================================
#
# kf(T) and kb(T) depend only on temperature, not on ODE state (coverages
# or gas concentrations).  Computing them here — once, before any ODE
# integration — means the inner-loop ODE solver only reads a matrix row
# instead of re-running quad integrations and BEP calculations on every call.
#
# kf_matrix[t_idx, j] = forward  rate constant of step j at T_ARR[t_idx]
# kb_matrix[t_idx, j] = backward rate constant of step j at T_ARR[t_idx]

print("\n── Precomputing rate constants over T_ARR ──")

kf_matrix, kb_matrix = precompute_kinetics(
    T_ARR, step_Keq, kin_params,
    nasa_gas, surf_data, cp_poly, STEPS,
)

print(f"  kf_matrix shape: {kf_matrix.shape}  (n_temps × n_steps)")
print(f"  kb_matrix shape: {kb_matrix.shape}  (n_temps × n_steps)")
print("  Precomputation complete.")


# ==============================================================================
# STEP 8 — MICROKINETICS: SWEEP OVER TEMPERATURE ARRAY
# ==============================================================================
#
# Warm-start strategy (Fix 4 + warm-start):
#   • First temperature: clean surface + ideal-gas feed via build_initial_conditions()
#   • Every subsequent temperature: surface coverages inherited from the
#     previous solution; only gas-phase concentrations are recalculated at
#     the new T via build_warm_start().
#
# This eliminates the violent adsorption transient that occurs when starting
# from a clean surface, reducing the stiffness seen by the solver at each
# temperature step from O(10^15) to O(10^5-10^8).

print("\n── Microkinetics sweep ──")
print(f"  {'T (K)':>8}  {'N2 conv.':>10}  {'NH3 (mol/cm³)':>15}"
      f"  {'t_stop (s)':>12}  Status")
print("  " + "-" * 68)

# Storage for steady-state results across all temperatures
ss_results = []   # list of np.ndarray, one per T; each of length N_VARS

# Build the initial condition for the first temperature using the Langmuir
# pre-equilibration estimate.  This places the surface close to adsorption/
# desorption balance from t=0, removing the violent adsorption spike that
# was causing step-size collapse at 50 bar (Fix A).
# build_initial_conditions() is still called first to get the gas-phase
# concentrations right; the surface part is then overwritten by the
# equilibrated estimate using the rate constants at T_ARR[0].
y0 = build_equilibrated_initial_conditions(
    T_ARR[0],
    kf_matrix[0, :],
    kb_matrix[0, :],
)

for t_idx, T in enumerate(T_ARR):

    # ── Row lookup: pre-computed rate constants for this temperature ──────────
    kf_arr = kf_matrix[t_idx, :]
    kb_arr = kb_matrix[t_idx, :]

    # ── Feed concentrations at this temperature (Option B) ────────────────────
    # C_feed is temperature-dependent through the ideal gas law C = P/(R*T).
    # It is computed once per temperature here and passed into run_microkinetics
    # as a fixed array — the ODE solver reads it as a constant during the
    # integration at this temperature (isothermal CSTR assumption).
    C_total_feed = P_BAR / (R_CM3_BAR * T)       # total molar conc [mol/cm³]
    C_feed = np.array([
        X_N2_FEED  * C_total_feed,                # C_N2_feed  [mol/cm³]
        X_H2_FEED  * C_total_feed,                # C_H2_feed  [mol/cm³]
        X_NH3_FEED * C_total_feed,                # C_NH3_feed [mol/cm³] = 0
    ])

    # ── Integrate ODE to steady state ─────────────────────────────────────────
    sol = run_microkinetics(T, kf_arr, kb_arr, C_feed, y0)

    # ── Extract final state ────────────────────────────────────────────────────
    y_final = sol.y[:, -1]
    ss_results.append(y_final)

    # ── Console status line ───────────────────────────────────────────────────
    N2_feed       = C_feed[0]           # feed concentration at this T
    N2_final      = y_final[IDX_N2G]
    NH3_final     = y_final[IDX_NH3G]
    t_stop        = sol.t[-1]

    # CSTR conversion: (C_feed - C_outlet) / C_feed
    # C_outlet = y_final[IDX_N2G] at steady state
    N2_conversion = (N2_feed - N2_final) / N2_feed if N2_feed > 0 else 0.0

    # Distinguish between three outcomes:
    #   "SS"     — steady-state event fired (early termination, good)
    #   "OK"     — reached T_SPAN end without SS event (rare)
    #   "FAILED" — solver did not complete
    if not sol.success:
        status_str = f"FAILED: {sol.message}"
    elif sol.t_events[0].size > 0:
        status_str = "SS"     # steady-state event fired
    else:
        status_str = "OK"     # reached end of T_SPAN

    print(f"  {T:>8.1f}  {N2_conversion:>10.4f}  {NH3_final:>15.4e}"
          f"  {t_stop:>12.3e}  {status_str}")

    # ── Build warm-start for the next temperature ─────────────────────────────
    # On the last iteration there is no next temperature, so we guard with
    # the index check to avoid an IndexError.
    if t_idx < len(T_ARR) - 1:
        y0 = build_warm_start(y_final, T_ARR[t_idx + 1])

# Convert list of state vectors into a 2-D array: (n_temperatures × N_VARS)
ss_array = np.array(ss_results)   # shape: (len(T_ARR), N_VARS)


# ==============================================================================
# STEP 9 — MICROKINETICS SUMMARY REPORT
# ==============================================================================

print("\n── Steady-state surface coverages at T = 700 K ──")

# Find the index of T closest to 700 K for a sample report
T_report   = 700.0
t_idx_700  = np.argmin(np.abs(T_ARR - T_report))
y_700      = ss_array[t_idx_700]

# Species name → state vector index mapping for reporting
surface_species_report = [
    ("N2(T)",  IDX_N2T),
    ("H(T)",   IDX_HT),
    ("NH3(T)", IDX_NH3T),
    ("N(T)",   IDX_NT),
    ("NH(T)",  IDX_NHT),
    ("NH2(T)", IDX_NH2T),
    ("N2(S)",  IDX_N2S),
    ("H(S)",   IDX_HS),
    ("NH3(S)", IDX_NH3S),
    ("N(S)",   IDX_NS),
    ("NH(S)",  IDX_NHS),
    ("NH2(S)", IDX_NH2S),
    ("N(SL)",  IDX_NSL),
]

for sp_name, idx in surface_species_report:
    print(f"  {sp_name:<12}  {y_700[idx]:.4e}  mol/cm²")

print(f"\n  N2  (gas)  {y_700[IDX_N2G]:.4e}  mol/cm³")
print(f"  H2  (gas)  {y_700[IDX_H2G]:.4e}  mol/cm³")
print(f"  NH3 (gas)  {y_700[IDX_NH3G]:.4e}  mol/cm³")


# ==============================================================================
# STEP 10 — COMPUTE TOF ARRAY FROM STEADY-STATE RESULTS
# ==============================================================================
#
# TOF [mol NH3 / (mol_site · s)] is computed from the steady-state net rates
# of the two NH3 desorption steps:
#
#   Step 2 :  NH3(T) <=>  NH3 + *(T)   (terrace desorption)
#   Step 5 :  NH3(S) <=>  NH3 + *(S)   (step    desorption)
#
# At steady state we re-evaluate compute_net_rates() using the pre-computed
# kf/kb row for that temperature and the final ODE state vector.
# This is inexpensive — it is just 22 multiplications with no integration.
#
# TOF normalisation:
#   TOF = (r_net_2 + r_net_5) * ABYV / SDTOT
#
# where:
#   r_net  [mol/(cm²·s)]  — net rate per unit catalyst surface area
#   ABYV   [cm²/cm³]      — converts to per unit reactor volume
#   SDTOT  [mol/cm²]      — normalises by total site density
#   result [mol NH3 / (mol_site · s)]
#
# Note: step indices 2 and 5 in the STEPS list correspond to NH3 desorption
# from terrace and step sites respectively (see kinetics.py STEPS definition).

STEP_IDX_NH3_DESORB_TERRACE = 2   # NH3 + *(T) <=> NH3(T)  — step index in STEPS
STEP_IDX_NH3_DESORB_STEP    = 5   # NH3 + *(S) <=> NH3(S)  — step index in STEPS

print("\n── Computing TOF from steady-state net rates ──")

tof_arr = np.zeros(len(T_ARR))

for t_idx in range(len(T_ARR)):

    # Re-use the pre-computed rate constants for this temperature
    kf_arr_t = kf_matrix[t_idx, :]
    kb_arr_t = kb_matrix[t_idx, :]

    # Steady-state concentrations for this temperature
    y_ss = ss_array[t_idx]

    # Evaluate all 22 net rates at steady state — fast, no integration
    _, _, rnet = compute_net_rates(y_ss, kf_arr_t, kb_arr_t)

    # NH3 desorption is the reverse of adsorption in the STEPS convention.
    # r_net > 0 means net flow toward products (NH3 desorbing from surface).
    # The adsorption step is written as:  NH3(gas) + * → NH3(ads)
    # so a NEGATIVE r_net means NH3 is leaving the surface → production.
    # We take the negative here so that TOF is positive for NH3 production.
    r_nh3_terrace = -rnet[STEP_IDX_NH3_DESORB_TERRACE]
    r_nh3_step    = -rnet[STEP_IDX_NH3_DESORB_STEP]

    # Sum the two desorption channels, convert to per-site basis
    tof_arr[t_idx] = (r_nh3_terrace + r_nh3_step) * ABYV / SDTOT


# ==============================================================================
# STEP 11 — PLOT TOF AND SURFACE COVERAGES
# ==============================================================================

plot_tof(T_ARR, tof_arr)

plot_surface_coverage(T_ARR, ss_array)


# ==============================================================================
# STEP 12 — BUILD STEADY-STATE RATE MATRICES AND PLOT ALL REACTION RATES
# ==============================================================================
#
# For each temperature we re-evaluate compute_net_rates() at the steady-state
# concentrations stored in ss_array.  This produces three (n_temps × n_steps)
# matrices:
#
#   rf_matrix  [i, j]  = forward  rate of step j at T_ARR[i]  [mol/(cm²·s)]
#   rb_matrix  [i, j]  = backward rate of step j at T_ARR[i]  [mol/(cm²·s)]
#   rnet_matrix[i, j]  = net      rate of step j at T_ARR[i]  [mol/(cm²·s)]
#
# This loop is fast — compute_net_rates() is 22 multiplications with no
# integration; all expensive work was done in Steps 7 and 8.

print("\n── Building steady-state rate matrices ──")

n_temps_rates = len(T_ARR)
n_steps_rates = len(STEPS)

rf_matrix   = np.zeros((n_temps_rates, n_steps_rates))
rb_matrix   = np.zeros((n_temps_rates, n_steps_rates))
rnet_matrix = np.zeros((n_temps_rates, n_steps_rates))

for t_idx in range(n_temps_rates):

    # Pre-computed rate constants for this temperature (row lookup)
    kf_arr_t = kf_matrix[t_idx, :]
    kb_arr_t = kb_matrix[t_idx, :]

    # Steady-state state vector for this temperature
    y_ss = ss_array[t_idx]

    # Evaluate all 22 forward, backward, and net rates simultaneously
    rf_row, rb_row, rnet_row = compute_net_rates(y_ss, kf_arr_t, kb_arr_t)

    # Store each row into the corresponding matrix
    rf_matrix[t_idx, :]   = rf_row
    rb_matrix[t_idx, :]   = rb_row
    rnet_matrix[t_idx, :] = rnet_row

print(f"  rf_matrix shape:   {rf_matrix.shape}  (n_temps × n_steps)")
print(f"  rb_matrix shape:   {rb_matrix.shape}")
print(f"  rnet_matrix shape: {rnet_matrix.shape}")

plot_reaction_rates(T_ARR, rf_matrix, rb_matrix, rnet_matrix, STEPS)
