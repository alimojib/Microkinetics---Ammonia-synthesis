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
    CSV_PATH,
    USE_CSV_THERMO,
    T_ARR,
    T_CP,
    SDTOT,          # total site density — used for TOF normalisation
    SDEN_T,         # terrace site density [mol/cm²]
    SDEN_S,         # step    site density [mol/cm²]
    R_CM3_BAR,      # gas constant [cm³·bar/(mol·K)] — for C_feed calculation
    P_BAR,          # total pressure [bar]
    X_N2_FEED,      # feed mole fraction of N2
    X_H2_FEED,      # feed mole fraction of H2
    X_NH3_FEED,     # feed mole fraction of NH3
)

from data_io import (
    read_surface_data,
    read_nasa_data,
    read_nasa_surface_data,
    read_nasa_from_csv,
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
    compute_vacancies,
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
    plot_ss_time,
)

from seqsim_thermo import compute_seqsim_step_keq


# ==============================================================================
# STEP 1 — LOAD DATA
# ==============================================================================

kin_params = read_kinetic_params(EXCEL_PATH)

# surf_data (DFT Cp points) is only needed for the Cp plot and as a fallback
# in H_species/S_species.  When USE_CSV_THERMO=True all species are covered by
# NASA-7 polynomials so surf_data is never accessed; skip the Excel read.
if not USE_CSV_THERMO:
    surf_data = read_surface_data(EXCEL_PATH)

if USE_CSV_THERMO:
    # Use SeqSim (data_ammonia.csv) NASA-7 coefficients for both gas and surface
    # species.  Zero-strain corrections are applied inside read_nasa_from_csv so
    # the returned dicts are equivalent to what SeqSim computes at strain=0.
    nasa_gas, nasa_surf = read_nasa_from_csv(CSV_PATH)
else:
    # Use the original Excel (sciadv.abl6576) NASA-7 coefficients.
    nasa_gas  = read_nasa_data(EXCEL_PATH)
    nasa_surf = read_nasa_surface_data(EXCEL_PATH)


# ==============================================================================
# STEP 2 — FIT SURFACE Cp POLYNOMIALS AND PLOT
# ==============================================================================
#
# When NASA-7 coefficients cover all surface species (USE_CSV_THERMO or
# nasa_surf from Excel), the Cp-integration path is never reached.
# In that case we skip loading surf_data, fitting cp_poly, and the Cp plot.

if USE_CSV_THERMO:
    surf_data = {}
    cp_poly   = {}
else:
    cp_poly = fit_surface_cp_polynomials(surf_data)
    plot_cp_fits(surf_data, cp_poly, nasa_gas, T_CP)


# ==============================================================================
# STEP 3 — COMPUTE Keq FOR EACH ELEMENTARY STEP AND PLOT
# ==============================================================================

step_Keq = compute_step_keq(STEPS, T_ARR, nasa_gas, surf_data, cp_poly,
                             nasa_surf=nasa_surf)

# SeqSim comparison traces: only meaningful when the main code uses a different
# thermodynamic source (Excel).  When USE_CSV_THERMO=True the two datasets are
# identical so the overlay is redundant and suppressed.
if USE_CSV_THERMO:
    thermo_label        = "NASA-7 (CSV)"
    step_Keq_seqsim     = None
    Keq_overall_seqsim  = None
else:
    thermo_label = "NASA-7 (Excel)"
    step_Keq_seqsim, Keq_overall_seqsim = compute_seqsim_step_keq(T_ARR)

plot_elementary_keq(STEPS, step_Keq, T_ARR,
                    thermo_label=thermo_label,
                    step_Keq_seqsim=step_Keq_seqsim)


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



# ==============================================================================
# STEP 6 — COMPUTE AND PLOT OVERALL Keq VS LITERATURE
# ==============================================================================

Keq_overall = compute_overall_keq(STEPS, step_Keq, nu_vec)
Keq_lit     = np.array([K_lit(T) for T in T_ARR])

plot_overall_keq(T_ARR, Keq_overall, Keq_lit,
                 thermo_label=thermo_label,
                 Keq_overall_seqsim=Keq_overall_seqsim)


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

kf_matrix, kb_matrix = precompute_kinetics(
    T_ARR, step_Keq, kin_params,
    nasa_gas, surf_data, cp_poly, STEPS,
    nasa_surf=nasa_surf,
)


# ==============================================================================
# DEBUG — A values and ΔH_rxn at T = 320 °C (593.15 K)
# ==============================================================================

T_DEBUG = 593.15   # K  (320 °C)
t_idx_debug = int(np.argmin(np.abs(T_ARR - T_DEBUG)))
T_actual = T_ARR[t_idx_debug]

print(f"\n── Debug: kinetic parameters at T = {T_actual:.2f} K ({T_actual - 273.15:.1f} °C) ──")

# ── Pre-exponential factors A for each surface reaction step ─────────────────
print("\n  Pre-exponential A values (from Excel, reaction-specific):")
print(f"  {'Step label':<44}  {'A (1/s)':>14}  {'alpha':>8}  {'E0 (kcal/mol)':>14}")
print("  " + "-" * 85)
for step in STEPS:
    if step.label in kin_params:
        p = kin_params[step.label]
        A_val  = p["A"]  if p["A"]  is not None else 0.0
        alpha  = p["alpha"]
        E0     = p["E0"]
        print(f"  {step.label:<44}  {A_val:>14.4e}  {alpha:>8.4f}  {E0:>14.4f}")

# ── ΔH_rxn for N2 dissociation steps ─────────────────────────────────────────
from microkinetics import _compute_dH_rxn

N2_DISS_LABELS = [
    "N2(T)+*(T)<=>2N(T)",       # step 11
    "N2(S)+*(SL)<=>N(S)+N(SL)", # step 12
    "N2(S)+*(S)<=>2N(S)",       # step 13
]

print(f"\n  ΔH_rxn for N2 dissociation steps at T = {T_actual:.2f} K:")
print(f"  {'Step label':<44}  {'ΔH_rxn (kcal/mol)':>18}  {'Ea_BEP (kcal/mol)':>18}")
print("  " + "-" * 85)
for step in STEPS:
    if step.label in N2_DISS_LABELS:
        dH = _compute_dH_rxn(step.species, T_actual, nasa_gas, surf_data, cp_poly,
                              nasa_surf)
        if step.label in kin_params:
            p  = kin_params[step.label]
            Ea = max(p["alpha"] * dH + p["E0"], 0.0)
        else:
            Ea = float("nan")
        print(f"  {step.label:<44}  {dH:>18.4f}  {Ea:>18.4f}")


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
ss_results    = []   # list of np.ndarray, one per T; each of length N_VARS
t_stop_arr    = np.zeros(len(T_ARR))   # simulated seconds when ODE stopped
ss_reached_arr = np.zeros(len(T_ARR), dtype=bool)  # True = SS event fired

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

    # ── Record stop time and whether the SS event actually fired ──────────────
    t_stop      = sol.t[-1]
    ss_fired    = sol.success and sol.t_events[0].size > 0
    t_stop_arr[t_idx]     = t_stop
    ss_reached_arr[t_idx] = ss_fired

    # ── Console status line ───────────────────────────────────────────────────
    N2_feed       = C_feed[0]           # feed concentration at this T
    N2_final      = y_final[IDX_N2G]
    NH3_final     = y_final[IDX_NH3G]

    # CSTR conversion: (C_feed - C_outlet) / C_feed
    # C_outlet = y_final[IDX_N2G] at steady state
    N2_conversion = (N2_feed - N2_final) / N2_feed if N2_feed > 0 else 0.0

    # Distinguish between three outcomes:
    #   "SS"     — steady-state event fired (early termination, good)
    #   "OK"     — reached T_SPAN end without SS event (rare)
    #   "FAILED" — solver did not complete
    if not sol.success:
        status_str = f"FAILED: {sol.message}"
    elif ss_fired:
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

# ── Coverage debug at T ≈ 320 °C (593.15 K) ──────────────────────────────────
y_dbg = ss_array[t_idx_debug]
vac_T_dbg, vac_S_dbg = compute_vacancies(y_dbg)

print(f"\n── Debug: Steady-state coverages at T = {T_actual:.2f} K ({T_actual-273.15:.1f} °C) ──")
print(f"  {'Species':<10}  {'mol/cm²':>14}  {'θ (fraction)':>14}")
print("  " + "-" * 42)
cov_species = [
    ("N2(T)",  y_dbg[IDX_N2T],  SDEN_T),
    ("N(T)",   y_dbg[IDX_NT],   SDEN_T),
    ("H(T)",   y_dbg[IDX_HT],   SDEN_T),
    ("NH3(T)", y_dbg[IDX_NH3T], SDEN_T),
    ("NH2(T)", y_dbg[IDX_NH2T], SDEN_T),
    ("NH(T)",  y_dbg[IDX_NHT],  SDEN_T),
    ("*(T)",   vac_T_dbg,       SDEN_T),
    ("N2(S)",  y_dbg[IDX_N2S],  SDEN_S),
    ("N(S)",   y_dbg[IDX_NS],   SDEN_S),
    ("H(S)",   y_dbg[IDX_HS],   SDEN_S),
    ("NH3(S)", y_dbg[IDX_NH3S], SDEN_S),
    ("NH2(S)", y_dbg[IDX_NH2S], SDEN_S),
    ("NH(S)",  y_dbg[IDX_NHS],  SDEN_S),
    ("N(SL)",  y_dbg[IDX_NSL],  SDEN_S),
    ("*(S)",   vac_S_dbg,       SDEN_S),
]
for sp_name, conc_cm2, sden in cov_species:
    theta = conc_cm2 / sden if sden > 0 else 0.0
    print(f"  {sp_name:<10}  {conc_cm2:>14.4e}  {theta:>14.6f}")
print(f"  {'N2 (gas)':<10}  {y_dbg[IDX_N2G]:>14.4e}  mol/cm³")
print(f"  {'H2 (gas)':<10}  {y_dbg[IDX_H2G]:>14.4e}  mol/cm³")
print(f"  {'NH3(gas)':<10}  {y_dbg[IDX_NH3G]:>14.4e}  mol/cm³")

# ── Keq for all adsorption and N2-dissociation steps at 593K ─────────────────
print(f"\n── Debug: Keq for adsorption/dissociation steps at T = {T_actual:.2f} K ──")
print(f"  {'Step':>4}  {'Label':<44}  {'Keq':>14}  {'kf':>14}  {'kb':>14}")
print("  " + "-" * 95)
for i_s, step in enumerate(STEPS):
    if i_s in (0, 1, 2, 3, 4, 5, 11, 12, 13):   # adsorption + N2 dissociation
        keq_val = step_Keq[i_s][t_idx_debug]
        kf_val  = kf_matrix[t_idx_debug, i_s]
        kb_val  = kb_matrix[t_idx_debug, i_s]
        print(f"  {i_s:>4}  {step.label:<44}  {keq_val:>14.4e}  {kf_val:>14.4e}  {kb_val:>14.4e}")


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
#   TOF = (r_net_2 + r_net_5) / SDTOT
#
# where:
#   r_net  [mol/(cm²·s)]  — net rate per unit catalyst surface area
#   SDTOT  [mol/cm²]      — total site density
#   result [mol NH3 / (mol_site · s)]  =  1/s
#
# Note: step indices 2 and 5 in the STEPS list correspond to NH3 desorption
# from terrace and step sites respectively (see kinetics.py STEPS definition).

STEP_IDX_NH3_DESORB_TERRACE = 2   # NH3 + *(T) <=> NH3(T)  — step index in STEPS
STEP_IDX_NH3_DESORB_STEP    = 5   # NH3 + *(S) <=> NH3(S)  — step index in STEPS

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

    # Sum the two desorption channels, normalise by site density
    tof_arr[t_idx] = (r_nh3_terrace + r_nh3_step) / SDTOT


# ==============================================================================
# STEP 11 — PLOT TOF AND SURFACE COVERAGES
# ==============================================================================

plot_tof(T_ARR, tof_arr, t_stop_arr, ss_reached_arr)

plot_ss_time(T_ARR, t_stop_arr, ss_reached_arr)

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

plot_reaction_rates(T_ARR, rf_matrix, rb_matrix, rnet_matrix, STEPS)
