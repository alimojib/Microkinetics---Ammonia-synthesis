"""
microkinetics.py
================
Power-law microkinetic model for the NH3 synthesis mechanism.

Rate constant expressions used
-------------------------------

  Adsorption steps (steps 0–5, Hertz-Knudsen):
    kf = (S / SDTOT) * sqrt(R_SI * T / (2π * MW))  [cm³/(mol·s)]
         × exp(-Ea / (R_CAL * T))
    with S = STICKING_COEFF, Ea = 0 for all adsorption steps.
    beta = 0 (no T^beta factor for adsorption steps).

  Surface reaction steps (steps 11–13, 16–21, Arrhenius + BEP):
    kf = A * T^beta * exp(-Ea / (R_CAL * T))
    with beta = 1.
    Ea(T) = alpha * dH_rxn(T) + E0   [kcal/mol]  — temperature-dependent
    through the reaction enthalpy dH_rxn(T) computed from thermodynamics.

  Diffusion / transfer steps (steps 6–10, 14–15):
    A = 0  →  kf = 0, kb = 0  (ignored at this stage).

  Backward rate constant (all steps):
    kb = kf / Keq

Elementary reaction steps
--------------------------
  Terrace adsorption (Hertz-Knudsen, Ea=0):
     0  N2  + *(T)  <=>  N2(T)
     1  H2  + 2*(T) <=>  2H(T)
     2  NH3 + *(T)  <=>  NH3(T)

  Step adsorption (Hertz-Knudsen, Ea=0):
     3  N2  + *(S)  <=>  N2(S)
     4  H2  + 2*(S) <=>  2H(S)
     5  NH3 + *(S)  <=>  NH3(S)

  Terrace ↔ step transfer (diffusion, A=0 — inactive):
     6  N(T)   + *(S) <=>  N(S)   + *(T)
     7  NH(T)  + *(S) <=>  NH(S)  + *(T)
     8  NH2(T) + *(S) <=>  NH2(S) + *(T)
     9  NH3(T) + *(S) <=>  NH3(S) + *(T)
    10  H(T)   + *(S) <=>  H(S)   + *(T)

  N2 dissociation:
    11  N2(T)  + *(T)  <=>  2N(T)              [Arrhenius+BEP]
    12  N2(S)  + *(SL) <=>  N(S)  + N(SL)      [Arrhenius+BEP]
    13  N2(S)  + *(S)  <=>  2N(S)              [Arrhenius+BEP]
    14  N2(S)  + *(T)  <=>  N(S)  + N(T)       [diffusion, A=0 — inactive]
    15  N(T)   + *(SL) <=>  N(SL) + *(T)       [diffusion, A=0 — inactive]

  Terrace hydrogenation (Arrhenius+BEP):
    16  N(T)   + H(T)  <=>  NH(T)  + *(T)
    17  NH(T)  + H(T)  <=>  NH2(T) + *(T)
    18  NH2(T) + H(T)  <=>  NH3(T) + *(T)

  Step hydrogenation (Arrhenius+BEP):
    19  N(S)   + H(S)  <=>  NH(S)  + *(S)
    20  NH(S)  + H(S)  <=>  NH2(S) + *(S)
    21  NH2(S) + H(S)  <=>  NH3(S) + *(S)

State vector layout (16 variables)
------------------------------------
  Index  Species      Units       Site
  -----  -------      -----       ----
    0    N2(T)        mol/cm²     terrace
    1    H(T)         mol/cm²     terrace
    2    NH3(T)       mol/cm²     terrace
    3    N(T)         mol/cm²     terrace
    4    NH(T)        mol/cm²     terrace
    5    NH2(T)       mol/cm²     terrace
    6    N2(S)        mol/cm²     step
    7    H(S)         mol/cm²     step
    8    NH3(S)       mol/cm²     step
    9    N(S)         mol/cm²     step
   10    NH(S)        mol/cm²     step
   11    NH2(S)       mol/cm²     step
   12    N(SL)        mol/cm²     lower-step
   13    N2           mol/cm³     gas
   14    H2           mol/cm³     gas
   15    NH3          mol/cm³     gas

Public API
----------
    precompute_kinetics(T_arr, step_Keq,
                        kin_params, nasa_gas,
                        surf_data, cp_poly, steps)
                                              -> (kf_matrix, kb_matrix)
                                                 shape: (n_temps, n_steps) each
    build_equilibrated_initial_conditions(T,
                        kf_arr, kb_arr)       -> np.ndarray  (y0, length 16)
    build_warm_start(y_prev, T_next)          -> np.ndarray  (y0, length 16)
    compute_rate_constants(T, step_Keq_T,
                           kin_params,
                           nasa_gas, surf_data, cp_poly, steps)
                                              -> (kf_arr, kb_arr)
    odes_rhs(t, y, kf_arr, kb_arr, C_feed) -> np.ndarray  (dy/dt, length 16)
    run_microkinetics(T, kf_arr, kb_arr,
                      C_feed, y0)          -> OdeResult
"""

import math
import numpy as np
from scipy.integrate import solve_ivp

from config import (
    R_CAL,          # gas constant  [cal/(mol·K)]
    R_SI,           # gas constant  [J/(mol·K)] = [Pa·m³/(mol·K)]
    R_CM3_BAR,      # gas constant  [cm³·bar/(mol·K)]
    P_BAR,          # total pressure [bar]
    SDTOT,          # total site density [mol/cm²]
    SDEN_T,         # terrace site density [mol/cm²]
    SDEN_S,         # step site density [mol/cm²]
    ABYV,           # catalyst area / reactor volume [cm²/cm³]
    STICKING_COEFF, # sticking coefficient [-]
    EA_ADS,         # adsorption activation energy [kcal/mol]
    MW,             # molecular weights dict [g/mol]
    T_SPAN,         # ODE integration time span [s]
    ODE_METHOD,     # ODE solver method ("Radau" — robust for stiff systems)
    ODE_RTOL,       # ODE relative tolerance
    ODE_ATOL_SURF,  # ODE absolute tolerance for surface coverage variables [mol/cm²]
    ODE_ATOL_GAS,   # ODE absolute tolerance for gas concentration variables [mol/cm³]
    ODE_MAX_STEP,   # ODE maximum step size [s]
    EPS_VAC,        # minimum vacant-site floor to avoid Jacobian kink [mol/cm²]
    SS_TOL,         # steady-state detection tolerance [mol/(cm²·s)]
    Q_IN,           # CSTR inlet  volumetric flow rate [cm³/s]
    Q_OUT,          # CSTR outlet volumetric flow rate [cm³/s]
    V_REACTOR,      # CSTR reactor volume [cm³]
    X_N2_FEED,      # feed mole fraction of N2  [-]
    X_H2_FEED,      # feed mole fraction of H2  [-]
    X_NH3_FEED,     # feed mole fraction of NH3 [-]
    USE_UNIFORM_A,  # if True, override all surface A with A_UNIFORM
    A_UNIFORM,      # uniform pre-exponential [s⁻¹] used when USE_UNIFORM_A=True
    EA_N_DIFF,      # N   inter-site diffusion Ea [kcal/mol] (SeqSim, zero strain)
    EA_NH_DIFF,     # NH  inter-site diffusion Ea [kcal/mol]
    EA_NH2_DIFF,    # NH2 inter-site diffusion Ea [kcal/mol]
    EA_NH3_DIFF,    # NH3 inter-site diffusion Ea [kcal/mol]
)

from thermodynamics import H_species


# ==============================================================================
# STATE VECTOR INDEXING
# ==============================================================================
#
# These integer constants give each species a fixed slot in the ODE
# state vector y.  Defining them as named constants makes the rate
# expressions below readable and immune to accidental index shifts.

# ── Terrace surface species (stored as mol/cm²) ───────────────────────────────
IDX_N2T  = 0    # N2(T)
IDX_HT   = 1    # H(T)
IDX_NH3T = 2    # NH3(T)
IDX_NT   = 3    # N(T)
IDX_NHT  = 4    # NH(T)
IDX_NH2T = 5    # NH2(T)

# ── Step surface species (stored as mol/cm²) ─────────────────────────────────
IDX_N2S  = 6    # N2(S)
IDX_HS   = 7    # H(S)
IDX_NH3S = 8    # NH3(S)
IDX_NS   = 9    # N(S)
IDX_NHS  = 10   # NH(S)
IDX_NH2S = 11   # NH2(S)
IDX_NSL  = 12   # N(SL)

# ── Gas-phase species (stored as mol/cm³) ────────────────────────────────────
IDX_N2G  = 13   # N2  gas
IDX_H2G  = 14   # H2  gas
IDX_NH3G = 15   # NH3 gas

# Total number of ODE variables
N_VARS = 16


# ==============================================================================
# VACANT SITE CONCENTRATIONS
# ==============================================================================

def compute_vacancies(y):
    """
    Compute the molar concentrations of vacant surface sites [mol/cm²]
    from the current state vector.

    The site balance on each site type states that the sum of all
    occupied coverages plus the vacant-site coverage equals the total
    site density for that type.

    Terrace:
        vac_T  = SDEN_T * ABYV - (N2_T + N_T + H_T + NH3_T + NH2_T + NH_T)

    Step:
        vac_S  = SDEN_S * ABYV - (N2_S + N_S + H_S + NH3_S + NH2_S + NH_S
                                   + N_SL)

    Note: SDEN * ABYV converts site density [mol/cm²] to volumetric
    units [mol/cm³_reactor] consistent with the gas-phase concentrations.
    The individual surface coverages y[i] are also in mol/cm² here, so
    they are used directly; the ABYV factor on SDEN accounts for the
    area-to-volume conversion of the total sites only.

    Parameters
    ----------
    y : np.ndarray of length N_VARS — current state vector

    Returns
    -------
    vac_T  : float — vacant terrace site concentration [mol/cm²]
    vac_S  : float — vacant step   site concentration [mol/cm²]
    """
    # Terrace: subtract all terrace adsorbates from total terrace site density
    vac_T = (
        SDEN_T
        - y[IDX_N2T]
        - y[IDX_NT]
        - y[IDX_HT]
        - y[IDX_NH3T]
        - y[IDX_NH2T]
        - y[IDX_NHT]
    )

    # Step: subtract all step and lower-step adsorbates from step site density
    vac_S = (
        SDEN_S
        - y[IDX_N2S]
        - y[IDX_NS]
        - y[IDX_HS]
        - y[IDX_NH3S]
        - y[IDX_NH2S]
        - y[IDX_NHS]
        - y[IDX_NSL]
    )

    # ── Smooth floor — replace hard max(..., 0) with EPS_VAC ─────────────────
    # A hard clip at exactly 0.0 creates a kink (non-differentiability) in the
    # ODE right-hand side.  Implicit solvers (Radau, LSODA) internally estimate
    # a Jacobian by finite differences; a kink at zero makes that estimate
    # unreliable, causing the step-size controller to collapse to machine
    # epsilon and then give up.
    # EPS_VAC (1e-30 mol/cm²) is physically indistinguishable from zero — it
    # is ~10^20 times smaller than SDEN_T — but it keeps the function smooth
    # so the Jacobian estimate stays well-conditioned.
    vac_T = max(vac_T, EPS_VAC)
    vac_S = max(vac_S, EPS_VAC)

    return vac_T, vac_S


# ==============================================================================
# RATE CONSTANT CALCULATIONS
# ==============================================================================

def _kf_adsorption(gas_species, T, dissociative=False):
    """
    Forward rate constant for one adsorption step via the Hertz-Knudsen
    expression.

    Molecular adsorption  (N2, NH3):  rate = kf × C_gas × vac
        kf = (S / SDTOT) × v_thermal          [cm³/(mol·s)]

    Dissociative adsorption (H2→2H):  rate = kf × C_gas × vac²
        kf = (S / SDTOT²) × v_thermal         [cm⁵/(mol²·s)]

    The extra 1/SDTOT for dissociative steps keeps the rate in mol/(cm²·s)
    when the second surface site is multiplied in as vac [mol/cm²].
    This matches Code 3's convention (Stick / (abyv × SDTOT²) × v_thermal)
    after accounting for Code 3 storing surface concentrations in mol/cm³.

    Parameters
    ----------
    gas_species   : str   — "N2", "H2", or "NH3"
    T             : float — temperature [K]
    dissociative  : bool  — True for H2+2*→2H (second-order in vacancies)
    """
    MW_kg = MW[gas_species] / 1000.0   # g/mol → kg/mol

    thermal_velocity_ms  = math.sqrt(R_SI * T / (2.0 * math.pi * MW_kg))
    thermal_velocity_cms = thermal_velocity_ms * 100.0

    arrhenius = math.exp(-EA_ADS * 1000.0 / (R_CAL * T))

    kf = (STICKING_COEFF / SDTOT) * thermal_velocity_cms * arrhenius

    # For dissociative adsorption the rate expression has vac² [mol/cm²]²;
    # one extra 1/SDTOT restores the correct mol/(cm²·s) dimensions.
    if dissociative:
        kf /= SDTOT

    return kf


def _kf_surface_reaction(step_label, T, dH_rxn_kcal, kin_params):
    """
    Forward rate constant for a surface reaction step via Arrhenius
    with a BEP-derived activation energy:

        Ea(T)  = alpha * dH_rxn(T) + E0       [kcal/mol]
        kf     = A * T^beta * exp(-Ea(T) * 1000 / (R_CAL * T))

    where beta = 1 (as decided) and A = alpha_1 from the Excel sheet.

    A floor of 0 kcal/mol is applied to Ea: if the BEP relationship
    predicts a negative activation energy (strongly exothermic step),
    the barrier is set to zero.  Negative activation energies are
    unphysical in the Arrhenius framework.

    Parameters
    ----------
    step_label    : str   — canonical step label matching kinetics.STEPS
    T             : float — temperature [K]
    dH_rxn_kcal   : float — reaction enthalpy ΔH [kcal/mol] at temperature T
    kin_params    : dict  — from data_io.read_kinetic_params()

    Returns
    -------
    float — kf;  0.0 if step_label not found in kin_params (zero-A steps)
    """
    # If this step is not in kin_params, it is a diffusion/transfer step
    # with A = 0; return zero immediately.
    if step_label not in kin_params:
        return 0.0

    params = kin_params[step_label]

    alpha = params["alpha"]
    E0    = params["E0"]      # kcal/mol
    A     = params["A"]       # pre-exponential, 1/s (from Excel)

    # Override with the uniform value when the comparison flag is set.
    # The Excel-derived A is still available in params["A"] at any time.
    if USE_UNIFORM_A:
        A = A_UNIFORM

    # If A is None (parsing failed) or effectively zero, short-circuit
    if A is None or A == 0.0:
        return 0.0

    # BEP activation energy [kcal/mol] — temperature-dependent through dH
    Ea_kcal = alpha * dH_rxn_kcal + E0

    # Apply physical floor: activation energy cannot be negative
    Ea_kcal = max(Ea_kcal, 0.0)

    # Convert kcal/mol → cal/mol for consistency with R_CAL [cal/(mol·K)]
    Ea_cal = Ea_kcal * 1000.0

    # Arrhenius pre-exponential × T^beta (beta = 1)
    beta = 1
    prefactor = A * (T ** beta)

    # Full rate constant
    kf = prefactor * math.exp(-Ea_cal / (R_CAL * T))

    return kf


def compute_rate_constants(T, step_Keq_T, kin_params,
                           nasa_gas, surf_data, cp_poly, steps,
                           nasa_surf=None):
    """
    Compute the forward (kf) and backward (kb) rate constants for all
    22 elementary steps at a single temperature T.

    Classification of each step:
        Steps 0–5   : adsorption → Hertz-Knudsen kf, kb = kf / Keq
        Steps 6–10  : diffusion/transfer → kf = 0, kb = 0
        Step  11    : N2(T)+*(T)<=>2N(T) → Arrhenius+BEP
        Step  12    : N2(S)+*(SL)<=>N(S)+N(SL) → Arrhenius+BEP
        Step  13    : N2(S)+*(S)<=>2N(S) → Arrhenius+BEP
        Step  14    : N2(S)+*(T)<=>N(S)+N(T) → diffusion → kf = 0
        Step  15    : N(T)+*(SL)<=>N(SL)+*(T) → diffusion → kf = 0
        Steps 16–21 : hydrogenation → Arrhenius+BEP

    The step index follows the STEPS list order defined in kinetics.py.

    Parameters
    ----------
    T           : float         — temperature [K]
    step_Keq_T  : list[float]   — Keq at temperature T for each step,
                                  i.e. step_Keq_T[i] = step_Keq[i][T_index]
    kin_params  : dict          — from data_io.read_kinetic_params()
    nasa_gas    : dict          — from data_io.read_nasa_data()
    surf_data   : dict          — from data_io.read_surface_data()
    cp_poly     : dict          — from thermodynamics.fit_surface_cp_polynomials()
    steps       : list[Step]    — from kinetics.STEPS

    Returns
    -------
    kf_arr : np.ndarray of length 22 — forward rate constants
    kb_arr : np.ndarray of length 22 — backward rate constants
    """
    n_steps = len(steps)
    kf_arr  = np.zeros(n_steps)
    kb_arr  = np.zeros(n_steps)

    # ── Indices of the 6 adsorption steps ─────────────────────────────────────
    # These are the first 6 steps in STEPS; ordered as:
    #   0: N2+*(T)<=>N2(T)
    #   1: H2+2*(T)<=>2H(T)   ← dissociative
    #   2: NH3+*(T)<=>NH3(T)
    #   3: N2+*(S)<=>N2(S)
    #   4: H2+2*(S)<=>2H(S)   ← dissociative
    #   5: NH3+*(S)<=>NH3(S)
    ADSORPTION_STEP_INDICES = [0, 1, 2, 3, 4, 5]

    # Map from step index → which gas species adsorbs in that step
    ADSORPTION_GAS_SPECIES = {
        0: "N2",
        1: "H2",
        2: "NH3",
        3: "N2",
        4: "H2",
        5: "NH3",
    }

    # H2 adsorption steps are dissociative (H2 + 2* → 2H*); their kf needs
    # an extra 1/SDTOT so the rate kf×C_H2×vac² stays in mol/(cm²·s).
    DISSOCIATIVE_STEP_INDICES = {1, 4}

    # ── Inter-site diffusion steps: fixed-Ea Arrhenius (SeqSim parameters) ──────
    # Ea values from SequentialSimulation.py at zero strain.
    # A = A_UNIFORM = 1.56e19 for all active diffusion steps (same as SeqSim).
    # kf [cm²/(mol·s)] = A × T × exp(-Ea×1000 / (R_CAL×T))
    # (ABYV factor cancels vs SeqSim convention where species are in mol/cm³.)
    # Step 10 (H diffusion) — inactive in SeqSim (step 15 commented out).
    # Step 13 (N2(S)+*(S)<=>2N(S)) — explicitly zeroed in SeqSim (step 8 off).
    # Step 14 (cross-site N2 dissociation) — inactive in both codes.
    DIFFUSION_STEP_EA = {
        6:  EA_N_DIFF,     # N(T)+*(S)  <=> N(S)+*(T)   — SeqSim step 14
        7:  EA_NH_DIFF,    # NH(T)+*(S) <=> NH(S)+*(T)  — SeqSim step 18
        8:  EA_NH2_DIFF,   # NH2(T)+*(S)<=> NH2(S)+*(T) — SeqSim step 17
        9:  EA_NH3_DIFF,   # NH3(T)+*(S)<=> NH3(S)+*(T) — SeqSim step 16
        15: EA_N_DIFF,     # N(T)+*(SL) <=> N(SL)+*(T)  — SeqSim step 21
    }

    # Steps kept at kf = 0 to match SeqSim:
    #   10 — H diffusion (SeqSim step 15 off)
    #   13 — N2(S)+*(S)<=>2N(S) (SeqSim step 8 off)
    #   14 — cross-site N2 dissociation (inactive in both)
    ZERO_RATE_STEP_INDICES = {10, 13, 14}

    for i, step in enumerate(steps):

        # ── Adsorption steps: Hertz-Knudsen ───────────────────────────────────
        if i in ADSORPTION_STEP_INDICES:
            gas_sp  = ADSORPTION_GAS_SPECIES[i]
            kf_val  = _kf_adsorption(gas_sp, T,
                                     dissociative=(i in DISSOCIATIVE_STEP_INDICES))

        # ── Diffusion/transfer steps: zero rate ───────────────────────────────
        elif i in ZERO_RATE_STEP_INDICES:
            kf_val = 0.0

        # ── Inter-site diffusion steps: fixed-Ea Arrhenius ────────────────────
        elif i in DIFFUSION_STEP_EA:
            Ea_cal = DIFFUSION_STEP_EA[i] * 1000.0   # kcal/mol → cal/mol
            kf_val = A_UNIFORM * (T ** 1) * math.exp(-Ea_cal / (R_CAL * T))

        # ── Surface reaction steps: Arrhenius + BEP ───────────────────────────
        else:
            # Compute the reaction enthalpy ΔH at temperature T.
            # This makes the BEP activation energy temperature-dependent.
            dH_rxn_kcal = _compute_dH_rxn(step.species, T,
                                           nasa_gas, surf_data, cp_poly,
                                           nasa_surf)

            kf_val = _kf_surface_reaction(step.label, T,
                                          dH_rxn_kcal, kin_params)

        # ── Backward rate: kb = kf / Kc ───────────────────────────────────────
        # step_Keq_T[i] is Kp — the dimensionless thermodynamic equilibrium
        # constant (activities: P/P° for gas, θ for surface, P° = 1 bar).
        #
        # For adsorption steps the rate expression mixes gas concentrations
        # [mol/cm³] with surface concentrations [mol/cm²].  In this mixed
        # unit system the concentration-based equilibrium constant Kc carries
        # units of cm³/mol, and Kp must be converted:
        #
        #   Kc = Kp × R·T / P°   [cm³/mol]   (P° = 1 bar, standard state)
        #
        # For pure surface steps every species lives in mol/cm², so the units
        # cancel and Kc = Kp (dimensionless) — no correction needed.
        #
        # Without this correction kb = kf/Kp is too large by a factor R·T/P°
        # ≈ 49 300 cm³/mol at 593 K, making desorption ~49 000× too fast and
        # suppressing adsorbed-H coverage by ~√(49 000) ≈ 220×.  Three
        # consecutive hydrogenation steps each requiring H(T) then compound
        # the error to ~220³ ≈ 10⁷ — explaining the observed TOF discrepancy.
        keq_val = step_Keq_T[i]   # Kp (dimensionless)

        if i in ADSORPTION_STEP_INDICES:
            # Convert Kp → Kc for the mixed gas/surface unit system.
            # R_CM3_BAR × T has units cm³·bar/mol; dividing by P° = 1 bar
            # gives cm³/mol, which is the correct unit for Kc here.
            keq_val = keq_val * R_CM3_BAR * T   # Kc [cm³/mol]

        if keq_val > 0.0 and kf_val > 0.0:
            kb_val = kf_val / keq_val
        else:
            kb_val = 0.0

        kf_arr[i] = kf_val
        kb_arr[i] = kb_val

    return kf_arr, kb_arr


def _compute_dH_rxn(sp_list, T, nasa_gas, surf_data, cp_poly, nasa_surf=None):
    """
    Compute ΔH_rxn [kcal/mol] for a single elementary step at
    temperature T using the unified H_species dispatcher from
    thermodynamics.py.

        ΔH = Σᵢ νᵢ · H_species(speciesᵢ, T)

    Parameters
    ----------
    sp_list   : list of (nu, species_name) — from Step.species
    T         : float — temperature [K]
    nasa_gas  : dict
    surf_data : dict
    cp_poly   : dict
    nasa_surf : dict|None — when provided, surface species H is from NASA-7

    Returns
    -------
    float — ΔH in kcal/mol
    """
    dH = sum(
        nu * H_species(sp, T, nasa_gas, surf_data, cp_poly, nasa_surf)
        for nu, sp in sp_list
    )
    return dH


# ==============================================================================
# POWER-LAW RATES FOR EACH ELEMENTARY STEP
# ==============================================================================

def compute_net_rates(y, kf_arr, kb_arr):
    """
    Compute the net reaction rate [mol/(cm²·s) or mol/(cm³·s)] for
    each of the 22 elementary steps using power-law kinetics.

    Power-law rate expressions
    --------------------------
    For each step:
        rf = kf * ∏(concentration of each reactant species ^ |ν|)
        rb = kb * ∏(concentration of each product  species ^ |ν|)
        rnet = rf - rb

    Concentrations used:
        Gas species      → y[IDX_*G]  [mol/cm³]
        Surface species  → y[IDX_*]   [mol/cm²]
        Vacant sites     → computed from site balance [mol/cm²]

    The mixed units (some mol/cm³, some mol/cm²) are intentional: the
    ODE right-hand side accounts for the ABYV conversion when updating
    gas-phase concentrations.

    Parameters
    ----------
    y      : np.ndarray length N_VARS — current state vector
    kf_arr : np.ndarray length 22     — forward rate constants
    kb_arr : np.ndarray length 22     — backward rate constants

    Returns
    -------
    rf   : np.ndarray length 22 — forward  rates
    rb   : np.ndarray length 22 — backward rates
    rnet : np.ndarray length 22 — net rates (rf - rb)
    """
    # Compute vacant site concentrations from the site balance
    vac_T, vac_S = compute_vacancies(y)

    # Convenience aliases — makes rate expressions below human-readable
    N2g  = y[IDX_N2G]
    H2g  = y[IDX_H2G]
    NH3g = y[IDX_NH3G]

    N2T  = y[IDX_N2T]
    HT   = y[IDX_HT]
    NH3T = y[IDX_NH3T]
    NT   = y[IDX_NT]
    NHT  = y[IDX_NHT]
    NH2T = y[IDX_NH2T]

    N2S  = y[IDX_N2S]
    HS   = y[IDX_HS]
    NH3S = y[IDX_NH3S]
    NS   = y[IDX_NS]
    NHS  = y[IDX_NHS]
    NH2S = y[IDX_NH2S]
    NSL  = y[IDX_NSL]

    # vac_T and vac_S computed above from site balance

    # ── All 22 forward and backward rates ────────────────────────────────────
    #
    # Each rate is written as a product of reactant concentrations for rf
    # and product concentrations for rb.
    #
    # Step numbering matches kinetics.py STEPS list order exactly.

    # Step 0:  N2 + *(T)  <=>  N2(T)
    rf_0  = kf_arr[0]  * N2g  * vac_T
    rb_0  = kb_arr[0]  * N2T

    # Step 1:  H2 + 2*(T)  <=>  2H(T)
    rf_1  = kf_arr[1]  * H2g  * vac_T**2
    rb_1  = kb_arr[1]  * HT**2

    # Step 2:  NH3 + *(T)  <=>  NH3(T)
    rf_2  = kf_arr[2]  * NH3g * vac_T
    rb_2  = kb_arr[2]  * NH3T

    # Step 3:  N2 + *(S)  <=>  N2(S)
    rf_3  = kf_arr[3]  * N2g  * vac_S
    rb_3  = kb_arr[3]  * N2S

    # Step 4:  H2 + 2*(S)  <=>  2H(S)
    rf_4  = kf_arr[4]  * H2g  * vac_S**2
    rb_4  = kb_arr[4]  * HS**2

    # Step 5:  NH3 + *(S)  <=>  NH3(S)
    rf_5  = kf_arr[5]  * NH3g * vac_S
    rb_5  = kb_arr[5]  * NH3S

    # Step 6:  N(T) + *(S)  <=>  N(S) + *(T)   [diffusion — zero rate]
    rf_6  = kf_arr[6]  * NT   * vac_S
    rb_6  = kb_arr[6]  * NS   * vac_T

    # Step 7:  NH(T) + *(S)  <=>  NH(S) + *(T)  [diffusion — zero rate]
    rf_7  = kf_arr[7]  * NHT  * vac_S
    rb_7  = kb_arr[7]  * NHS  * vac_T

    # Step 8:  NH2(T) + *(S)  <=>  NH2(S) + *(T)  [diffusion — zero rate]
    rf_8  = kf_arr[8]  * NH2T * vac_S
    rb_8  = kb_arr[8]  * NH2S * vac_T

    # Step 9:  NH3(T) + *(S)  <=>  NH3(S) + *(T)  [diffusion — zero rate]
    rf_9  = kf_arr[9]  * NH3T * vac_S
    rb_9  = kb_arr[9]  * NH3S * vac_T

    # Step 10: H(T) + *(S)  <=>  H(S) + *(T)  [diffusion — zero rate]
    rf_10 = kf_arr[10] * HT   * vac_S
    rb_10 = kb_arr[10] * HS   * vac_T

    # Step 11: N2(T) + *(T)  <=>  2N(T)
    rf_11 = kf_arr[11] * N2T  * vac_T
    rb_11 = kb_arr[11] * NT**2

    # Step 12: N2(S) + *(SL)  <=>  N(S) + N(SL)
    #          vac_SL is not tracked separately; it is approximated as
    #          vac_S here (step sites are a single pool at this level).
    vac_SL = vac_S   # placeholder — extend when SL sites tracked separately
    rf_12 = kf_arr[12] * N2S  * vac_SL
    rb_12 = kb_arr[12] * NS   * NSL

    # Step 13: N2(S) + *(S)  <=>  2N(S)
    rf_13 = kf_arr[13] * N2S  * vac_S
    rb_13 = kb_arr[13] * NS**2

    # Step 14: N2(S) + *(T)  <=>  N(S) + N(T)  [diffusion — zero rate]
    rf_14 = kf_arr[14] * N2S  * vac_T
    rb_14 = kb_arr[14] * NS   * NT

    # Step 15: N(T) + *(SL)  <=>  N(SL) + *(T)  [diffusion — zero rate]
    rf_15 = kf_arr[15] * NT   * vac_SL
    rb_15 = kb_arr[15] * NSL  * vac_T

    # Step 16: N(T) + H(T)  <=>  NH(T) + *(T)
    rf_16 = kf_arr[16] * NT   * HT
    rb_16 = kb_arr[16] * NHT  * vac_T

    # Step 17: NH(T) + H(T)  <=>  NH2(T) + *(T)
    rf_17 = kf_arr[17] * NHT  * HT
    rb_17 = kb_arr[17] * NH2T * vac_T

    # Step 18: NH2(T) + H(T)  <=>  NH3(T) + *(T)
    rf_18 = kf_arr[18] * NH2T * HT
    rb_18 = kb_arr[18] * NH3T * vac_T

    # Step 19: N(S) + H(S)  <=>  NH(S) + *(S)
    rf_19 = kf_arr[19] * NS   * HS
    rb_19 = kb_arr[19] * NHS  * vac_S

    # Step 20: NH(S) + H(S)  <=>  NH2(S) + *(S)
    rf_20 = kf_arr[20] * NHS  * HS
    rb_20 = kb_arr[20] * NH2S * vac_S

    # Step 21: NH2(S) + H(S)  <=>  NH3(S) + *(S)
    rf_21 = kf_arr[21] * NH2S * HS
    rb_21 = kb_arr[21] * NH3S * vac_S

    # Pack all rates into arrays for clean downstream handling
    rf   = np.array([rf_0,  rf_1,  rf_2,  rf_3,  rf_4,  rf_5,
                     rf_6,  rf_7,  rf_8,  rf_9,  rf_10, rf_11,
                     rf_12, rf_13, rf_14, rf_15, rf_16, rf_17,
                     rf_18, rf_19, rf_20, rf_21])

    rb   = np.array([rb_0,  rb_1,  rb_2,  rb_3,  rb_4,  rb_5,
                     rb_6,  rb_7,  rb_8,  rb_9,  rb_10, rb_11,
                     rb_12, rb_13, rb_14, rb_15, rb_16, rb_17,
                     rb_18, rb_19, rb_20, rb_21])

    rnet = rf - rb

    return rf, rb, rnet


# ==============================================================================
# ODE RIGHT-HAND SIDE
# ==============================================================================

def odes_rhs(t, y, kf_arr, kb_arr, C_feed):
    """
    Compute dy/dt for the coupled surface-coverage / gas-concentration
    ODE system at a single time point t — CSTR formulation.

    The system has 16 equations:
        • 13 surface species (mol/cm²)  — d(coverage)/dt = net rate
        • 3  gas-phase species (mol/cm³) — d(concentration)/dt  [CSTR]

    Surface ODEs
    ------------
    Surface species are not convected — they stay on the catalyst:

        d(y_i)/dt = Σ_j  ν_ij * rnet_j     [mol/(cm²·s)]

    Gas-phase ODEs — CSTR balance
    ------------------------------
    Each gas species has three contributions:

        d(C_i)/dt = (reaction term)
                  + (Q_IN / V_REACTOR) * C_i_feed   ← inlet brings fresh feed
                  - (Q_OUT / V_REACTOR) * C_i        ← outlet removes current gas

    where:
        reaction term  = Σ_j  ν_ij * rnet_j * ABYV  [mol/(cm³·s)]
        Q_IN           = 0.76 cm³/s  (H2 34.2 sccm + N2 11.4 sccm)
        Q_OUT          = Q_IN        (isobaric, constant volume)
        V_REACTOR      = 7.7 cm³
        C_i_feed       = feed concentration at reactor T [mol/cm³]

    At steady state with no reaction this gives C_i → C_i_feed, which
    is the correct CSTR limit.  With reaction the gas-phase concentrations
    settle at a balance between consumption, production, and flow — unlike
    the batch model where reactants deplete to zero over time.

    Parameters
    ----------
    t      : float        — current time [s]  (required by scipy interface)
    y      : np.ndarray   — current state vector, length N_VARS
    kf_arr : np.ndarray   — forward  rate constants for all 22 steps
    kb_arr : np.ndarray   — backward rate constants for all 22 steps
    C_feed : np.ndarray   — feed concentrations [mol/cm³], length 3,
                            ordered [C_N2_feed, C_H2_feed, C_NH3_feed].
                            Pre-computed in main.py at each temperature
                            from the ideal gas law and feed mole fractions.

    Returns
    -------
    dydt : np.ndarray of length N_VARS — time derivatives
    """
    # Compute all net reaction rates from the current state
    _, _, rnet = compute_net_rates(y, kf_arr, kb_arr)

    # Unpack net rates for readability
    (r0,  r1,  r2,  r3,  r4,  r5,
     r6,  r7,  r8,  r9,  r10, r11,
     r12, r13, r14, r15, r16, r17,
     r18, r19, r20, r21) = rnet

    # Unpack feed concentrations
    C_N2_feed  = C_feed[0]   # mol/cm³
    C_H2_feed  = C_feed[1]   # mol/cm³
    C_NH3_feed = C_feed[2]   # mol/cm³

    # Pre-compute the flow rate / volume ratio — used in all three gas ODEs
    # tau_inv = Q/V  [1/s] — reciprocal of the mean residence time
    tau_inv = Q_IN / V_REACTOR   # = Q_OUT / V_REACTOR (isobaric assumption)

    # Initialise derivative vector with zeros
    dydt = np.zeros(N_VARS)

    # ── Surface species ODEs (unchanged from batch — no flow terms) ───────────
    # Each line adds the stoichiometric contribution of every step that
    # involves the species.  Positive ν = produced; negative ν = consumed.

    # d[N2(T)]/dt  — step 0: N2+*(T)⇌N2(T);  step 11: N2(T)+*(T)⇌2N(T)
    dydt[IDX_N2T]  = (+r0) + (-r11)

    # d[H(T)]/dt   — step 1: H2+2*(T)⇌2H(T);  steps 16,17,18: hydrogenation;
    #                step 10: H(T)+*(S)⇌H(S)+*(T)
    dydt[IDX_HT]   = (+2*r1) + (-r16) + (-r17) + (-r18) + (+r10)

    # d[NH3(T)]/dt — step 2: NH3+*(T)⇌NH3(T);  step 18: NH2(T)+H(T)⇌NH3(T)+*(T);
    #                step 9: NH3(T)+*(S)⇌NH3(S)+*(T) (transfer away from terrace)
    dydt[IDX_NH3T] = (+r2) + (+r18) + (-r9)

    # d[N(T)]/dt   — step 11: N2(T)+*(T)⇌2N(T);  step 14: N2(S)+*(T)⇌N(S)+N(T);
    #                step 6: N(T)+*(S)⇌N(S)+*(T);  step 15: N(T)+*(SL)⇌N(SL)+*(T);
    #                step 16: N(T)+H(T)⇌NH(T)+*(T)
    dydt[IDX_NT]   = (+2*r11) + (+r14) + (-r6) + (-r15) + (-r16)

    # d[NH(T)]/dt  — step 16: N(T)+H(T)⇌NH(T)+*(T);
    #                step 7: NH(T)+*(S)⇌NH(S)+*(T);  step 17: NH(T)+H(T)⇌NH2(T)+*(T)
    dydt[IDX_NHT]  = (+r16) + (-r7) + (-r17)

    # d[NH2(T)]/dt — step 17: NH(T)+H(T)⇌NH2(T)+*(T);
    #                step 8: NH2(T)+*(S)⇌NH2(S)+*(T);  step 18: NH2(T)+H(T)⇌NH3(T)+*(T)
    dydt[IDX_NH2T] = (+r17) + (-r8) + (-r18)

    # d[N2(S)]/dt  — step 3: N2+*(S)⇌N2(S);
    #                steps 12,13,14: N2(S) dissociation pathways
    dydt[IDX_N2S]  = (+r3) + (-r12) + (-r13) + (-r14)

    # d[H(S)]/dt   — step 4: H2+2*(S)⇌2H(S);  step 10: H(T)+*(S)⇌H(S)+*(T);
    #                steps 19,20,21: step hydrogenation consumes H(S)
    dydt[IDX_HS]   = (+2*r4) + (-r10) + (-r19) + (-r20) + (-r21)

    # d[NH3(S)]/dt — step 5: NH3+*(S)⇌NH3(S);  step 21: NH2(S)+H(S)⇌NH3(S)+*(S);
    #                step 9: NH3(T)+*(S)⇌NH3(S)+*(T) (transfer in from terrace)
    dydt[IDX_NH3S] = (+r5) + (+r21) + (+r9)

    # d[N(S)]/dt   — steps 6,12,13,14: produce N(S);  step 19: N(S)+H(S)⇌NH(S)+*(S)
    dydt[IDX_NS]   = (+r6) + (+r12) + (+r13) + (+r14) + (-r19)

    # d[NH(S)]/dt  — step 7: NH(T)+*(S)⇌NH(S)+*(T);
    #                step 19: N(S)+H(S)⇌NH(S)+*(S);  step 20: NH(S)+H(S)⇌NH2(S)+*(S)
    dydt[IDX_NHS]  = (+r7) + (+r19) + (-r20)

    # d[NH2(S)]/dt — step 8: NH2(T)+*(S)⇌NH2(S)+*(T);
    #                step 20: NH(S)+H(S)⇌NH2(S)+*(S);  step 21: NH2(S)+H(S)⇌NH3(S)+*(S)
    dydt[IDX_NH2S] = (+r8) + (+r20) + (-r21)

    # d[N(SL)]/dt  — step 12: N2(S)+*(SL)⇌N(S)+N(SL);
    #                step 15: N(T)+*(SL)⇌N(SL)+*(T)
    dydt[IDX_NSL]  = (+r12) + (+r15)

    # ── Gas-phase species ODEs — CSTR balance ─────────────────────────────────
    #
    # Three terms for each gas species:
    #   1. Reaction:  ν * rnet * ABYV   converts surface rate → volumetric rate
    #   2. Inlet:     tau_inv * C_feed   fresh reactants entering the reactor
    #   3. Outlet:    tau_inv * C_i      current contents leaving the reactor
    #
    # The sign convention on the reaction term:
    #   N2 and H2 are consumed by adsorption (steps 0,1,3,4) → negative
    #   NH3 is consumed by adsorption (steps 2,5) → negative
    #   (desorption is handled by the backward rates in rnet)

    # d[N2]/dt
    # r0, r3 are in mol/(cm²·s); ×ABYV converts to mol/(cm³·s) to match C_gas units.
    dydt[IDX_N2G] = (
        ((-r0) + (-r3)) * ABYV          # reaction: adsorption removes N2
        + tau_inv * C_N2_feed            # inlet: N2 flowing in
        - tau_inv * y[IDX_N2G]           # outlet: N2 flowing out
    )

    # d[H2]/dt
    dydt[IDX_H2G] = (
        ((-r1) + (-r4)) * ABYV          # reaction: adsorption removes H2
        + tau_inv * C_H2_feed            # inlet: H2 flowing in
        - tau_inv * y[IDX_H2G]           # outlet: H2 flowing out
    )

    # d[NH3]/dt
    # NH3 is produced by desorption (backward of steps 2 and 5) and
    # consumed by adsorption (forward of steps 2 and 5).
    # Both are already captured in the net rates rnet[2] and rnet[5].
    # The feed contains no NH3 (X_NH3_FEED = 0), so C_NH3_feed = 0,
    # meaning the inlet term vanishes and the outlet term only removes
    # whatever NH3 has been produced by the reaction.
    dydt[IDX_NH3G] = (
        ((-r2) + (-r5)) * ABYV          # reaction: net NH3 adsorption/desorption
        + tau_inv * C_NH3_feed           # inlet: NH3 in feed (= 0 for this case)
        - tau_inv * y[IDX_NH3G]          # outlet: NH3 produced by reaction exits
    )

    return dydt


# ==============================================================================
# PRECOMPUTATION OF RATE CONSTANTS OVER THE FULL TEMPERATURE ARRAY
# ==============================================================================

def precompute_kinetics(T_arr, step_Keq, kin_params,
                        nasa_gas, surf_data, cp_poly, steps,
                        nasa_surf=None):
    """
    Compute kf and kb for every elementary step at every temperature in
    T_arr and store the results in two 2-D matrices.

    Why this function exists
    ------------------------
    kf(T) and kb(T) depend only on temperature — not on surface coverages
    or gas concentrations.  They are therefore constants with respect to
    the ODE state vector y.  Pre-computing them once and reading a row
    during the ODE loop avoids repeating:
        • scipy.integrate.quad calls (H/S integration for surface species)
        • BEP activation energy calculations
        • Hertz-Knudsen square-root evaluations
    for every temperature point in T_arr.  At 200 temperature points the
    savings are large; inside a stiff ODE with thousands of internal steps
    they would be enormous.

    Layout of the output matrices
    ------------------------------
    Both matrices have shape  (n_temps, n_steps)  where:
        n_temps = len(T_arr)
        n_steps = len(steps)   (22 elementary steps)

    Row i corresponds to T_arr[i].
    Column j corresponds to steps[j].

    To get the rate-constant arrays for a single temperature index t_idx:
        kf_arr = kf_matrix[t_idx, :]
        kb_arr = kb_matrix[t_idx, :]

    Parameters
    ----------
    T_arr     : np.ndarray    — temperature array [K], shape (n_temps,)
    step_Keq  : list[np.ndarray]
                              — equilibrium constants; step_Keq[j][i] is
                                Keq for step j at temperature T_arr[i].
                                Produced by thermodynamics.compute_step_keq().
    kin_params : dict         — from data_io.read_kinetic_params()
    nasa_gas   : dict         — from data_io.read_nasa_data()
    surf_data  : dict         — from data_io.read_surface_data()
    cp_poly    : dict         — from thermodynamics.fit_surface_cp_polynomials()
    steps      : list[Step]   — from kinetics.STEPS

    Returns
    -------
    kf_matrix : np.ndarray, shape (n_temps, n_steps)
        kf_matrix[i, j] = forward rate constant of step j at T_arr[i]

    kb_matrix : np.ndarray, shape (n_temps, n_steps)
        kb_matrix[i, j] = backward rate constant of step j at T_arr[i]
    """
    n_temps = len(T_arr)
    n_steps = len(steps)

    # Allocate output matrices up front — filling row-by-row is efficient
    kf_matrix = np.zeros((n_temps, n_steps))
    kb_matrix = np.zeros((n_temps, n_steps))

    for t_idx, T in enumerate(T_arr):

        # Extract the scalar Keq for every step at this temperature index.
        # step_Keq[j] is a 1-D array over all temperatures; we pick element
        # t_idx to get the single float we need for this row.
        step_Keq_at_T = [step_Keq[j][t_idx] for j in range(n_steps)]

        # Delegate to the single-temperature function — all rate-constant
        # logic stays in one place, making it easy to change formulas later.
        kf_arr, kb_arr = compute_rate_constants(
            T, step_Keq_at_T, kin_params,
            nasa_gas, surf_data, cp_poly, steps,
            nasa_surf,
        )

        # Store the row for this temperature
        kf_matrix[t_idx, :] = kf_arr
        kb_matrix[t_idx, :] = kb_arr

    return kf_matrix, kb_matrix


# ==============================================================================
# INITIAL CONDITIONS
# ==============================================================================

def _enforce_min_vacancy(y0, min_vac_frac=0.1):
    """
    Rescale surface coverages so each site type retains at least min_vac_frac
    (default 10 %) vacant sites.

    When the ODE steady state is strongly N-poisoned, the warm-start carries
    θ_N ≈ 99% into the next temperature step.  With virtually no free sites
    available, the hydrogenation steps (N+H→NH) are starved of co-reactant
    and the solver takes extremely long to escape the poisoned state.

    Capping occupation at 1 - min_vac_frac (90%) by proportionally scaling
    all adsorbates on each site type is the minimal perturbation that restores
    enough vacant sites for the chemistry to proceed without introducing
    thermodynamic inconsistencies.

    Site type groupings
    -------------------
    Terrace (T) : N2(T), H(T), NH3(T), N(T), NH(T), NH2(T)   →  capacity SDEN_T
    Step (S+SL) : N2(S), H(S), NH3(S), N(S), NH(S), NH2(S),
                  N(SL)                                         →  capacity SDEN_S

    Parameters
    ----------
    y0           : np.ndarray — state vector (modified in-place and returned)
    min_vac_frac : float      — minimum vacant-site fraction [0, 1)

    Returns
    -------
    y0 : np.ndarray — same array, modified in-place
    """
    _TERRACE_IDX = [IDX_N2T, IDX_HT, IDX_NH3T, IDX_NT, IDX_NHT, IDX_NH2T]
    _STEP_IDX    = [IDX_N2S, IDX_HS, IDX_NH3S, IDX_NS, IDX_NHS, IDX_NH2S,
                    IDX_NSL]

    max_occ_T = (1.0 - min_vac_frac) * SDEN_T
    max_occ_S = (1.0 - min_vac_frac) * SDEN_S

    occ_T = sum(y0[i] for i in _TERRACE_IDX)
    if occ_T > max_occ_T > 0:
        scale = max_occ_T / occ_T
        for i in _TERRACE_IDX:
            y0[i] *= scale

    occ_S = sum(y0[i] for i in _STEP_IDX)
    if occ_S > max_occ_S > 0:
        scale = max_occ_S / occ_S
        for i in _STEP_IDX:
            y0[i] *= scale

    return y0

def build_initial_conditions(T):
    """
    Build the initial state vector y0 for the ODE system with a clean
    surface and ideal-gas feed concentrations.

    Used only for the very first temperature in the sweep.  All subsequent
    temperatures use build_warm_start() to inherit the previous steady-state
    surface coverages, and build_equilibrated_initial_conditions() is called
    before the first ODE integration to avoid the violent adsorption transient.

    Initial conditions:
        • All surface coverages = 0 (clean surface).
        • Gas-phase concentrations from ideal gas law at P_BAR,
          with feed composition N2:H2:NH3 = 1:3:0 (stoichiometric).

    Ideal gas:   C_i = (x_i * P_BAR) / (R_CM3_BAR * T)

    Parameters
    ----------
    T : float — temperature [K]

    Returns
    -------
    y0 : np.ndarray of length N_VARS
    """
    y0 = np.zeros(N_VARS)

    x_N2  = 0.25    # 1 part N2 out of 4 total (1 N2 + 3 H2)
    x_H2  = 0.75    # 3 parts H2
    x_NH3 = 0.0     # no NH3 in feed

    C_total = P_BAR / (R_CM3_BAR * T)

    y0[IDX_N2G]  = x_N2  * C_total
    y0[IDX_H2G]  = x_H2  * C_total
    y0[IDX_NH3G] = x_NH3 * C_total

    return y0


def build_equilibrated_initial_conditions(T, kf_arr, kb_arr):
    """
    Build a pre-equilibrated initial surface condition using a single-site
    Langmuir isotherm approximation for each site type (terrace and step).

    Why this function exists — Fix A
    ---------------------------------
    Starting from a clean surface at 50 bar causes an extremely violent
    adsorption transient at t=0: all 6 adsorption rate constants are
    large (Hertz-Knudsen), and gas-phase concentrations at 50 bar are
    50× higher than at 1 bar.  The resulting spike in dy/dt can exceed
    10^15 mol/(cm²·s), which forces the Radau solver to take steps of
    ~10^-20 s immediately — thousands of tiny steps before any meaningful
    chemistry occurs.

    A Langmuir pre-equilibration estimate places the surface much closer
    to the adsorption/desorption balance from the start, eliminating the
    spike entirely.

    Langmuir isotherm approximation
    ---------------------------------
    For each adsorbing species i on site type σ (terrace T or step S):

        K_ads_i = kf_i / kb_i        (adsorption equilibrium constant)
        n_i     = K_ads_i * C_i      (occupancy numerator)

    The competitive Langmuir fraction for species i on site σ is:

        θ_i = n_i / (1 + Σ_j n_j)

    where the sum runs over all species adsorbing on σ.

    Coverage is then:
        y[IDX_i] = θ_i * SDEN_σ

    This is an approximation because:
        1. It ignores lateral interactions between adsorbates.
        2. It assumes adsorption/desorption equilibrium (ignores surface
           reactions that consume/produce adsorbates).
        3. It uses the same Langmuir formula for dissociative adsorption
           (H2 → 2H) as for molecular adsorption, which is not exact.

    Despite these limitations it is far better than zero as a starting
    point and removes the dominant source of solver failure at high pressure.

    Parameters
    ----------
    T      : float       — temperature [K]
    kf_arr : np.ndarray  — forward  rate constants at temperature T
    kb_arr : np.ndarray  — backward rate constants at temperature T

    Returns
    -------
    y0 : np.ndarray of length N_VARS
    """
    y0 = np.zeros(N_VARS)

    # ── Gas-phase concentrations at feed conditions ───────────────────────────
    x_N2  = 0.25
    x_H2  = 0.75
    x_NH3 = 0.0

    C_total = P_BAR / (R_CM3_BAR * T)

    C_N2  = x_N2  * C_total
    C_H2  = x_H2  * C_total
    C_NH3 = x_NH3 * C_total

    y0[IDX_N2G]  = C_N2
    y0[IDX_H2G]  = C_H2
    y0[IDX_NH3G] = C_NH3

    # ── Langmuir pre-equilibration for terrace sites (steps 0, 1, 2) ─────────
    # Adsorption equilibrium constants for terrace adsorption steps.
    # Guard against kb = 0 (which would give infinite K_ads).
    K_N2T  = kf_arr[0] / kb_arr[0]  if kb_arr[0]  > 0 else 0.0  # step 0
    K_H2T  = kf_arr[1] / kb_arr[1]  if kb_arr[1]  > 0 else 0.0  # step 1
    K_NH3T = kf_arr[2] / kb_arr[2]  if kb_arr[2]  > 0 else 0.0  # step 2

    # Langmuir occupancy numerators
    n_N2T  = K_N2T  * C_N2
    n_H2T  = K_H2T  * C_H2
    n_NH3T = K_NH3T * C_NH3

    # Denominator: 1 + sum of all numerators (competitive adsorption)
    denom_T = 1.0 + n_N2T + n_H2T + n_NH3T

    # Fractional coverages — clip to [0, 1] for safety
    theta_N2T  = min(n_N2T  / denom_T, 1.0)
    theta_H2T  = min(n_H2T  / denom_T, 1.0)
    theta_NH3T = min(n_NH3T / denom_T, 1.0)

    # Convert dimensionless coverage to mol/cm²
    # H2 dissociates → 2H(T), so H coverage is 2 × H2 fractional coverage.
    # Cap HT at SDEN_T: the Langmuir formula (molecular, not dissociative) can
    # yield 2×theta_H2T = 2 when theta_H2T → 1 at low T, giving y0[HT] = 2×SDEN_T
    # which violates the site balance and forces vac_T to EPS_VAC from t=0,
    # stalling the ODE for >> T_SPAN seconds while H slowly desorbs.
    y0[IDX_N2T]  = theta_N2T  * SDEN_T
    y0[IDX_HT]   = min(2.0 * theta_H2T * SDEN_T, SDEN_T)
    y0[IDX_NH3T] = theta_NH3T * SDEN_T

    # ── Langmuir pre-equilibration for step sites (steps 3, 4, 5) ────────────
    K_N2S  = kf_arr[3] / kb_arr[3]  if kb_arr[3]  > 0 else 0.0  # step 3
    K_H2S  = kf_arr[4] / kb_arr[4]  if kb_arr[4]  > 0 else 0.0  # step 4
    K_NH3S = kf_arr[5] / kb_arr[5]  if kb_arr[5]  > 0 else 0.0  # step 5

    n_N2S  = K_N2S  * C_N2
    n_H2S  = K_H2S  * C_H2
    n_NH3S = K_NH3S * C_NH3

    denom_S = 1.0 + n_N2S + n_H2S + n_NH3S

    theta_N2S  = min(n_N2S  / denom_S, 1.0)
    theta_H2S  = min(n_H2S  / denom_S, 1.0)
    theta_NH3S = min(n_NH3S / denom_S, 1.0)

    y0[IDX_N2S]  = theta_N2S  * SDEN_S
    y0[IDX_HS]   = min(2.0 * theta_H2S * SDEN_S, SDEN_S)   # cap at site capacity
    y0[IDX_NH3S] = theta_NH3S * SDEN_S

    _enforce_min_vacancy(y0)
    return y0


def build_warm_start(y_prev, T_next):
    """
    Build the initial state vector for temperature T_next by inheriting
    the steady-state surface coverages from the previous temperature and
    updating only the gas-phase concentrations.

    Why this is important
    ---------------------
    Starting every temperature from a clean surface forces the solver to
    resolve the violent initial adsorption transient (all coverages rising
    from zero) at every single temperature point.  That transient is the
    hardest part of the ODE for a stiff solver.

    Instead, the surface coverages that were at steady state at T_prev
    are a physically sensible starting point for T_next — the surface
    does not suddenly become clean when temperature changes by 5 K.

    Gas-phase concentrations DO need to be recalculated at T_next because
    the ideal-gas law C = P/(R*T) makes them temperature-dependent:
        C(T_next) = C(T_prev) * T_prev / T_next

    All 13 surface coverage slots are copied directly from y_prev.
    The three gas-phase slots are recomputed from the ideal gas law at
    T_next with the same feed mole fractions as build_initial_conditions().

    Parameters
    ----------
    y_prev : np.ndarray of length N_VARS
        Steady-state (or final) solution from the previous temperature.
    T_next : float
        The next temperature point [K].

    Returns
    -------
    y0 : np.ndarray of length N_VARS
        Warm-start initial condition for T_next.
    """
    # Start from a copy of the previous solution so all surface slots are
    # inherited without having to list each index individually.
    y0 = y_prev.copy()

    # Recalculate gas-phase concentrations at the new temperature.
    # The feed composition (mole fractions) is unchanged — only T changes.
    x_N2  = 0.25   # same as in build_initial_conditions()
    x_H2  = 0.75
    x_NH3 = 0.0

    C_total = P_BAR / (R_CM3_BAR * T_next)

    y0[IDX_N2G]  = x_N2  * C_total
    y0[IDX_H2G]  = x_H2  * C_total
    y0[IDX_NH3G] = x_NH3 * C_total

    # Clip any surface coverages that drifted slightly negative during the
    # previous ODE integration — a warm start with negative values would
    # immediately corrupt the site balance for the next solve.
    for idx in range(IDX_N2G):   # indices 0-12 are all surface species
        if y0[idx] < 0.0:
            y0[idx] = 0.0

    _enforce_min_vacancy(y0)
    return y0


# ==============================================================================
# ODE SOLVER
# ==============================================================================

def run_microkinetics(T, kf_arr, kb_arr, C_feed, y0):
    """
    Integrate the microkinetic ODE system from t=0 until either T_SPAN[1]
    is reached or the steady-state termination event fires.

    Solver choice — Radau
    ----------------------
    "Radau" is an implicit Runge-Kutta method (order 5) designed for stiff
    systems.  It is more robust than LSODA on microkinetic problems where
    the stiffness ratio exceeds ~10^10.

    Steady-state termination event
    --------------------------------
    The integration is stopped early when the infinity-norm of dy/dt drops
    below SS_TOL.  Avoids integrating far past convergence.

    Parameters
    ----------
    T      : float       — temperature [K]
    kf_arr : np.ndarray  — forward  rate constants at temperature T
    kb_arr : np.ndarray  — backward rate constants at temperature T
    C_feed : np.ndarray  — feed concentrations [mol/cm³], length 3,
                           ordered [C_N2_feed, C_H2_feed, C_NH3_feed].
                           Passed to odes_rhs() as a closure variable so
                           the CSTR inlet term uses the correct T-dependent
                           feed concentrations without recomputing them
                           inside the hot ODE path.
    y0     : np.ndarray  — initial condition vector of length N_VARS

    Returns
    -------
    sol : scipy OdeResult object
        sol.t       — time points [s]
        sol.y       — state array (N_VARS × n_time_points)
        sol.success — True if integration completed or steady-state reached
        sol.message — solver status message
    """
    def rhs(t, y):
        """
        Thin wrapper so solve_ivp receives a 2-argument callable.
        kf_arr, kb_arr, and C_feed are captured from the enclosing scope.
        """
        return odes_rhs(t, y, kf_arr, kb_arr, C_feed)

    def steady_state_event(t, y):
        """
        Termination event: fires when max|dy/dt| falls below SS_TOL.
        Returns max|dy/dt| - SS_TOL; solve_ivp stops when this crosses
        zero from above (direction = -1).
        """
        dydt = odes_rhs(t, y, kf_arr, kb_arr, C_feed)
        return np.max(np.abs(dydt)) - SS_TOL

    steady_state_event.terminal  = True
    steady_state_event.direction = -1

    # Per-variable absolute tolerance vector
    atol_vec = np.full(N_VARS, ODE_ATOL_SURF)   # tight for surface species
    atol_vec[IDX_N2G]  = ODE_ATOL_GAS           # looser for gas species
    atol_vec[IDX_H2G]  = ODE_ATOL_GAS
    atol_vec[IDX_NH3G] = ODE_ATOL_GAS

    sol = solve_ivp(
        fun      = rhs,
        t_span   = T_SPAN,
        y0       = y0,
        method   = ODE_METHOD,
        rtol     = ODE_RTOL,
        atol     = atol_vec,
        max_step = ODE_MAX_STEP,
        events   = steady_state_event,
        dense_output = False,
    )

    return sol

