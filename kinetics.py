"""
kinetics.py
===========
Defines all 22 elementary steps for the NH3 synthesis mechanism and
provides the linear-algebra routines that combine them into the overall
equilibrium constant.

Responsibilities
----------------
1. Build the list of Step objects (reaction network definition)
2. Build the species-stoichiometry matrix A
3. Solve  A_active · ν = b  for the linear-combination multipliers
4. Compute the overall Keq as ∏ Keq_i^νᵢ
5. Provide the literature Keq reference K_lit(T)

Public API
----------
    STEPS                           list[Step]   — 22 elementary steps
    ALL_SPECIES                     list[str]    — ordered species list
    build_stoichiometry_matrix(...) -> np.ndarray
    solve_linear_combination(...)   -> (nu_vec, b_check, residual)
    compute_overall_keq(...)        -> np.ndarray
    K_lit(T)                        -> float
"""

import numpy as np

from thermodynamics import Step


# ==============================================================================
# ELEMENTARY STEPS DEFINITION
# ==============================================================================
#
# Each Step carries:
#   label   — human-readable reaction string
#   active  — whether this step enters the overall Keq linear combination
#   species — list of (stoichiometric_coefficient, species_name) tuples
#               negative coefficient = consumed (reactant)
#               positive coefficient = produced  (product)
#
# All 22 steps are defined here. 15 are active=True; 7 are active=False
# (their multipliers ν are forced to 0 in the overall Keq solve).
#
# Inactive steps (active=False):
#   • N(T)+*(S)<=>N(S)+*(T)       — terrace→step N transfer
#   • NH(T)+*(S)<=>NH(S)+*(T)     — terrace→step NH transfer
#   • NH2(T)+*(S)<=>NH2(S)+*(T)   — terrace→step NH2 transfer
#   • NH3(T)+*(S)<=>NH3(S)+*(T)   — terrace→step NH3 transfer
#   • H(T)+*(S)<=>H(S)+*(T)       — terrace→step H transfer
#   • N2(S)+*(T)<=>N(S)+N(T)      — cross-site N2 dissociation
#   • N(T)+*(SL)<=>N(SL)+*(T)     — terrace→lower-step N transfer
#
# The step "N(T)+*(SL)<=>N(SL)+*(T)" was added to close the N(SL) mass
# balance; it is retained in the list but deactivated.
#
# Three steps were added in revision 2:
#   • N2(S)+*(S)<=>2N(S)       — on-step N2 dissociation (both N stay on step)
#   • N2(S)+*(T)<=>N(S)+N(T)  — cross-site N2 dissociation (deactivated)
#   • H(T)+*(S)<=>H(S)+*(T)   — hydrogen terrace→step transfer (deactivated)

STEPS = [
    # ── Terrace adsorption ────────────────────────────────────────────────────
    Step(
        label   = "N2+*(T)<=>N2(T)",
        active  = True,
        species = [(-1, "N2"), (-1, "*T"), (+1, "N2(T)")],
    ),
    Step(
        label   = "H2+2*(T)<=>2H(T)",
        active  = True,
        species = [(-1, "H2"), (-2, "*T"), (+2, "H(T)")],
    ),
    Step(
        label   = "NH3+*(T)<=>NH3(T)",
        active  = True,
        species = [(-1, "NH3"), (-1, "*T"), (+1, "NH3(T)")],
    ),

    # ── Step adsorption ───────────────────────────────────────────────────────
    Step(
        label   = "N2+*(S)<=>N2(S)",
        active  = True,
        species = [(-1, "N2"), (-1, "*S"), (+1, "N2(S)")],
    ),
    Step(
        label   = "H2+2*(S)<=>2H(S)",
        active  = True,
        species = [(-1, "H2"), (-2, "*S"), (+2, "H(S)")],
    ),
    Step(
        label   = "NH3+*(S)<=>NH3(S)",
        active  = True,
        species = [(-1, "NH3"), (-1, "*S"), (+1, "NH3(S)")],
    ),

    # ── Terrace ↔ step transfer ───────────────────────────────────────────────
    Step(
        label   = "N(T)+*(S)<=>N(S)+*(T)",
        active  = False,
        species = [(-1, "N(T)"), (-1, "*S"), (+1, "N(S)"), (+1, "*T")],
    ),
    Step(
        label   = "NH(T)+*(S)<=>NH(S)+*(T)",
        active  = False,
        species = [(-1, "NH(T)"), (-1, "*S"), (+1, "NH(S)"), (+1, "*T")],
    ),
    Step(
        label   = "NH2(T)+*(S)<=>NH2(S)+*(T)",
        active  = False,
        species = [(-1, "NH2(T)"), (-1, "*S"), (+1, "NH2(S)"), (+1, "*T")],
    ),
    Step(
        label   = "NH3(T)+*(S)<=>NH3(S)+*(T)",
        active  = False,
        species = [(-1, "NH3(T)"), (-1, "*S"), (+1, "NH3(S)"), (+1, "*T")],
    ),

    # ── Hydrogen transfer — terrace to step ───────────────────────────────────
    # Added in revision 2.  Allows adsorbed H on the terrace to spill over to
    # a vacant step site, coupling the two H reservoirs directly.
    Step(
        label   = "H(T)+*(S)<=>H(S)+*(T)",
        active  = False,
        species = [(-1, "H(T)"), (-1, "*S"), (+1, "H(S)"), (+1, "*T")],
    ),

    # ── Terrace N2 dissociation ───────────────────────────────────────────────
    Step(
        label   = "N2(T)+*(T)<=>2N(T)",
        active  = True,
        species = [(-1, "N2(T)"), (-1, "*T"), (+2, "N(T)")],
    ),

    # ── Step N2 dissociation ──────────────────────────────────────────────────
    Step(
        label   = "N2(S)+*(SL)<=>N(S)+N(SL)",
        active  = True,
        species = [(-1, "N2(S)"), (-1, "*SL"), (+1, "N(S)"), (+1, "N(SL)")],
    ),

    # ── On-step N2 dissociation (both N atoms stay on step site) ─────────────
    # Added in revision 2.  Complements the existing step dissociation pathway
    # where one N lands on the lower-step site (SL).
    Step(
        label   = "N2(S)+*(S)<=>2N(S)",
        active  = True,
        species = [(-1, "N2(S)"), (-1, "*S"), (+2, "N(S)")],
    ),

    # ── Cross-site N2 dissociation (one N to terrace, one stays on step) ─────
    # Added in revision 2.  The adsorbed N2 on the step dissociates with a
    # vacant terrace site; one N atom migrates to the terrace.
    Step(
        label   = "N2(S)+*(T)<=>N(S)+N(T)",
        active  = False,
        species = [(-1, "N2(S)"), (-1, "*T"), (+1, "N(S)"), (+1, "N(T)")],
    ),

    # ── Terrace N ↔ lower-step transfer (closes N(SL) mass balance) ──────────
    Step(
        label   = "N(T)+*(SL)<=>N(SL)+*(T)",
        active  = False,
        species = [(-1, "N(T)"), (-1, "*SL"), (+1, "N(SL)"), (+1, "*T")],
    ),

    # ── Terrace hydrogenation ─────────────────────────────────────────────────
    Step(
        label   = "N(T)+H(T)<=>NH(T)+*(T)",
        active  = True,
        species = [(-1, "N(T)"), (-1, "H(T)"), (+1, "NH(T)"), (+1, "*T")],
    ),
    Step(
        label   = "NH(T)+H(T)<=>NH2(T)+*(T)",
        active  = True,
        species = [(-1, "NH(T)"), (-1, "H(T)"), (+1, "NH2(T)"), (+1, "*T")],
    ),
    Step(
        label   = "NH2(T)+H(T)<=>NH3(T)+*(T)",
        active  = True,
        species = [(-1, "NH2(T)"), (-1, "H(T)"), (+1, "NH3(T)"), (+1, "*T")],
    ),

    # ── Step hydrogenation ────────────────────────────────────────────────────
    Step(
        label   = "N(S)+H(S)<=>NH(S)+*(S)",
        active  = True,
        species = [(-1, "N(S)"), (-1, "H(S)"), (+1, "NH(S)"), (+1, "*S")],
    ),
    Step(
        label   = "NH(S)+H(S)<=>NH2(S)+*(S)",
        active  = True,
        species = [(-1, "NH(S)"), (-1, "H(S)"), (+1, "NH2(S)"), (+1, "*S")],
    ),
    Step(
        label   = "NH2(S)+H(S)<=>NH3(S)+*(S)",
        active  = True,
        species = [(-1, "NH2(S)"), (-1, "H(S)"), (+1, "NH3(S)"), (+1, "*S")],
    ),
]

# Convenience: total step count used elsewhere
N_STEPS = len(STEPS)


# ==============================================================================
# COMPLETE SPECIES LIST
# ==============================================================================
#
# Fixed ordering used to build the stoichiometry matrix.
# Gas-phase species (N2, H2, NH3) must appear here so their rows carry
# the correct target stoichiometry coefficients in the solve.

ALL_SPECIES = [
    # Surface intermediates — terrace
    "N2(T)", "H(T)", "NH3(T)", "N(T)", "NH(T)", "NH2(T)",
    # Surface intermediates — step / lower-step
    "N2(S)", "H(S)", "NH3(S)", "N(S)", "NH(S)", "NH2(S)", "N(SL)",
    # Vacant sites
    "*T", "*S", "*SL",
    # Gas-phase species
    "N2", "H2", "NH3",
]


# ==============================================================================
# STOICHIOMETRY MATRIX
# ==============================================================================

def build_stoichiometry_matrix(steps, all_species):
    """
    Build the full stoichiometry matrix A of shape (n_species × n_steps).

    A[i, j] is the stoichiometric coefficient of species i in step j.
    A value of 0 means species i does not appear in step j.

    Parameters
    ----------
    steps       : list[Step]  — elementary steps (STEPS from this module)
    all_species : list[str]   — ordered species list (ALL_SPECIES)

    Returns
    -------
    np.ndarray of shape (len(all_species), len(steps))
    """
    n_sp    = len(all_species)
    n_steps = len(steps)
    sp_idx  = {sp: i for i, sp in enumerate(all_species)}

    A = np.zeros((n_sp, n_steps))

    for j, step in enumerate(steps):
        for nu, sp in step.species:
            if sp in sp_idx:
                A[sp_idx[sp], j] += nu

    return A


# ==============================================================================
# LINEAR COMBINATION SOLVER
# ==============================================================================

def solve_linear_combination(A, steps, all_species):
    """
    Find multipliers ν such that A_active · ν_active = b, where:

        b  encodes the overall reaction N2 + 3H2 → 2NH3:
            N2  row : −1
            H2  row : −3
            NH3 row : +2
            all other rows : 0  (surface intermediates must cancel)

    Only the active step columns enter the solve; excluded steps have
    ν = 0 by definition.  np.linalg.lstsq is used to handle any
    numerical redundancy in the active column set.

    Parameters
    ----------
    A           : np.ndarray — full stoichiometry matrix from
                               build_stoichiometry_matrix()
    steps       : list[Step] — STEPS from this module
    all_species : list[str]  — ALL_SPECIES from this module

    Returns
    -------
    nu_vec   : np.ndarray of length len(steps)
        Full multiplier vector; excluded step positions are exactly 0.
    b_check  : np.ndarray of length len(all_species)
        A @ nu_vec — used to verify the net stoichiometry.
    residual : float
        max |A @ nu_vec − b|; should be ≈ 1e-14 for a consistent system.
    """
    n_sp    = len(all_species)
    n_steps = len(steps)
    sp_idx  = {sp: i for i, sp in enumerate(all_species)}

    # Column indices for active and excluded steps
    active_idx = [j for j, step in enumerate(steps) if step.active]

    # Target overall stoichiometry vector
    b = np.zeros(n_sp)
    b[sp_idx["N2"]]  = -1.0
    b[sp_idx["H2"]]  = -3.0
    b[sp_idx["NH3"]] = +2.0

    # Solve using only the active columns (least squares)
    A_active               = A[:, active_idx]
    nu_active, _, rank, sv = np.linalg.lstsq(A_active, b, rcond=None)

    # Place active-step solutions back into a full-length vector
    # (excluded entries remain 0)
    nu_vec = np.zeros(n_steps)
    for k, j in enumerate(active_idx):
        nu_vec[j] = nu_active[k]

    # Compute residual as a quality check
    b_check  = A @ nu_vec
    residual = np.max(np.abs(b_check - b))

    return nu_vec, b_check, residual


# ==============================================================================
# OVERALL Keq
# ==============================================================================

def compute_overall_keq(steps, step_Keq, nu_vec):
    """
    Compute the overall equilibrium constant as a product of powers of
    the elementary-step Keq values:

        Keq_overall(T) = ∏ᵢ  Keq_i(T)^νᵢ

    Only active steps contribute because inactive steps have νᵢ = 0
    (anything raised to the power 0 equals 1).

    Parameters
    ----------
    steps    : list[Step]       — STEPS from this module
    step_Keq : list[np.ndarray] — from thermodynamics.compute_step_keq()
    nu_vec   : np.ndarray       — multipliers from solve_linear_combination()

    Returns
    -------
    np.ndarray — Keq_overall at each temperature point
    """
    # Start from 1 (neutral element for multiplication)
    n_temps     = len(step_Keq[0])
    Keq_overall = np.ones(n_temps)

    active_idx = [j for j, step in enumerate(steps) if step.active]

    for j in active_idx:
        Keq_overall *= step_Keq[j] ** nu_vec[j]

    return Keq_overall


# ==============================================================================
# LITERATURE REFERENCE Keq
# ==============================================================================

def K_lit(T):
    """
    Literature equilibrium constant for N2 + 3H2 <=> 2NH3 at temperature T.

    Based on the standard Temkin–Pyzhev thermodynamic correlation.
    The base formula gives Keq for (1/2)N2 + (3/2)H2 <=> NH3; squaring
    converts it to the 2NH3 stoichiometry used throughout this project.

    Parameters
    ----------
    T : float — temperature in K

    Returns
    -------
    float — K_lit(T)
    """
    log10_K = (
        2.1
        + (1.0 / 4.571) * (9591.0 / T - 4.6e-4 * T + 8.5e-7 * T**2)
        - (4.98 / 1.985) * np.log10(T)
    )

    # Square to match the 2NH3 stoichiometry convention
    return (10**log10_K) ** 2
