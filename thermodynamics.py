"""
thermodynamics.py
=================
All thermodynamic calculations for the NH3 equilibrium project.

Responsibilities
----------------
1. NASA polynomial Cp, H, S for gas-phase species (NH3, N2, H2)
2. Degree-3 polynomial fits to surface-species Cp(T) data
3. H(T) and S(T) by numerical integration for surface species
4. A unified dispatcher — H_species() / S_species() — so callers
   never need to know whether a species is gas-phase or surface
5. The Step dataclass that replaces the raw (label, active, sp_list)
   tuple used in the original code
6. ΔG and Keq computation for every elementary step

Public API
----------
    fit_surface_cp_polynomials(surf_data)           -> dict[str, np.poly1d]
    H_species(name, T, nasa_gas, surf_data, cp_poly) -> float   [kcal/mol]
    S_species(name, T, nasa_gas, surf_data, cp_poly) -> float   [cal/mol/K]
    Step                                             dataclass
    compute_step_keq(steps, T_arr, nasa_gas,
                     surf_data, cp_poly)             -> list[np.ndarray]
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.integrate import quad

from config import R_CAL, T_CP, CP_POLY_DEGREE, VACANT_SITES


# ==============================================================================
# STEP DATACLASS
# ==============================================================================

@dataclass
class Step:
    """
    Represents one elementary reaction step.

    Attributes
    ----------
    label   : str
        Human-readable reaction string, e.g. "N2+*(T)<=>N2(T)".

    active  : bool
        True  → this step participates in the overall Keq linear combination.
        False → its multiplier is forced to 0 (excluded from the solve).

    species : list of (nu, name) tuples
        nu   (int or float) : stoichiometric coefficient
                              negative = reactant, positive = product
        name (str)          : species identifier matching the Excel sheet
                              or NASA polynomial dict keys

    Example
    -------
        Step(
            label   = "H2+2*(T)<=>2H(T)",
            active  = True,
            species = [(-1, "H2"), (-2, "*T"), (+2, "H(T)")],
        )
    """

    label   : str
    active  : bool
    species : List[Tuple[float, str]] = field(default_factory=list)


# ==============================================================================
# SURFACE Cp POLYNOMIAL FITS
# ==============================================================================

def fit_surface_cp_polynomials(surf_data):
    """
    Fit a polynomial of degree CP_POLY_DEGREE (from config) to the
    discrete Cp(T) data for each surface species.

    The resulting poly1d objects are later used in numerical integration
    to compute H(T) and S(T) for surface species.

    Parameters
    ----------
    surf_data : dict — from data_io.read_surface_data()
                       { name : {"H0": float, "S0": float, "Cp": array} }

    Returns
    -------
    dict
        { species_name : np.poly1d }
    """
    cp_poly = {}

    for name, d in surf_data.items():
        coeffs        = np.polyfit(T_CP, d["Cp"], CP_POLY_DEGREE)
        cp_poly[name] = np.poly1d(coeffs)

    return cp_poly


# ==============================================================================
# NASA POLYNOMIAL HELPERS
# ==============================================================================

def _nasa_coeffs(species, T, nasa_gas):
    """
    Select the correct set of NASA 7-coefficient polynomial coefficients
    for the given species at temperature T.

    The NASA polynomial is defined piecewise:
        T ≤ T_break → use the "low"  coefficient set
        T >  T_break → use the "high" coefficient set

    Parameters
    ----------
    species : str        — "NH3", "N2", or "H2"
    T       : float      — temperature in K
    nasa_gas : dict      — from data_io.read_nasa_data()

    Returns
    -------
    np.ndarray of shape (7,) — [a1, a2, a3, a4, a5, a6, a7]
    """
    d = nasa_gas[species]
    return d["low"] if T <= d["T_break"] else d["high"]


def nasa_Cp(species, T, nasa_gas):
    """
    Heat capacity Cp [cal/(mol·K)] from the NASA 7-coefficient polynomial.

    NASA polynomial form:
        Cp / R = a1 + a2*T + a3*T² + a4*T³ + a5*T⁴

    Parameters
    ----------
    species  : str   — "NH3", "N2", or "H2"
    T        : float — temperature in K
    nasa_gas : dict  — from data_io.read_nasa_data()

    Returns
    -------
    float — Cp in cal/(mol·K)
    """
    a = _nasa_coeffs(species, T, nasa_gas)

    Cp_over_R = (
        a[0]
        + a[1] * T
        + a[2] * T**2
        + a[3] * T**3
        + a[4] * T**4
    )

    return Cp_over_R * R_CAL   # convert dimensionless Cp/R → cal/(mol·K)


def nasa_H(species, T, nasa_gas):
    """
    Enthalpy H(T) [kcal/mol] from the NASA 7-coefficient polynomial.

    NASA polynomial form:
        H / (RT) = a1 + a2*T/2 + a3*T²/3 + a4*T³/4 + a5*T⁴/5 + a6/T

    The a6 coefficient encodes the reference-state enthalpy.

    Parameters
    ----------
    species  : str   — "NH3", "N2", or "H2"
    T        : float — temperature in K
    nasa_gas : dict  — from data_io.read_nasa_data()

    Returns
    -------
    float — H in kcal/mol
    """
    a = _nasa_coeffs(species, T, nasa_gas)

    H_over_RT = (
        a[0]
        + a[1] * T / 2.0
        + a[2] * T**2 / 3.0
        + a[3] * T**3 / 4.0
        + a[4] * T**4 / 5.0
        + a[5] / T
    )

    # H/RT is dimensionless; multiply by R*T to get cal/mol, then /1000 for kcal
    return H_over_RT * R_CAL * T / 1000.0


def nasa_S(species, T, nasa_gas):
    """
    Entropy S(T) [cal/(mol·K)] from the NASA 7-coefficient polynomial.

    NASA polynomial form:
        S / R = a1*ln(T) + a2*T + a3*T²/2 + a4*T³/3 + a5*T⁴/4 + a7

    The a7 coefficient encodes the reference-state entropy.

    Parameters
    ----------
    species  : str   — "NH3", "N2", or "H2"
    T        : float — temperature in K
    nasa_gas : dict  — from data_io.read_nasa_data()

    Returns
    -------
    float — S in cal/(mol·K)
    """
    a = _nasa_coeffs(species, T, nasa_gas)

    S_over_R = (
        a[0] * np.log(T)
        + a[1] * T
        + a[2] * T**2 / 2.0
        + a[3] * T**3 / 3.0
        + a[4] * T**4 / 4.0
        + a[6]
    )

    return S_over_R * R_CAL   # convert dimensionless S/R → cal/(mol·K)


# ==============================================================================
# SURFACE SPECIES H(T) AND S(T) BY INTEGRATION
# ==============================================================================

def _H_surf(name, T, surf_data, cp_poly):
    """
    Enthalpy H(T) [kcal/mol] for a surface species.

    Computed by integrating the fitted Cp polynomial from 0 K to T,
    then adding the DFT reference enthalpy at 0 K:

        H(T) = H(0 K) + ∫₀ᵀ Cp(T') dT'

    Parameters
    ----------
    name      : str        — surface species name
    T         : float      — temperature in K
    surf_data : dict       — from data_io.read_surface_data()
    cp_poly   : dict       — from fit_surface_cp_polynomials()

    Returns
    -------
    float — H in kcal/mol
    """
    # quad returns (integral_value, estimated_error)
    integral, _ = quad(cp_poly[name], 0, T)

    # integral is in cal/mol (Cp in cal/(mol·K) × dT in K); divide by 1000
    return surf_data[name]["H0"] + integral / 1000.0


def _S_surf(name, T, surf_data, cp_poly):
    """
    Entropy S(T) [cal/(mol·K)] for a surface species.

    Computed by integrating Cp(T')/T' from ≈0 K to T and adding the
    DFT reference entropy at 0 K:

        S(T) = S(0 K) + ∫₀ᵀ [Cp(T') / T'] dT'

    A small lower bound (eps = 1e-3 K) replaces 0 to avoid the 1/T
    singularity in the integrand at T' = 0.

    Parameters
    ----------
    name      : str        — surface species name
    T         : float      — temperature in K
    surf_data : dict       — from data_io.read_surface_data()
    cp_poly   : dict       — from fit_surface_cp_polynomials()

    Returns
    -------
    float — S in cal/(mol·K)
    """
    eps = 1e-3   # lower integration limit to avoid division by zero at T=0

    def integrand(Tp):
        return cp_poly[name](Tp) / Tp

    integral, _ = quad(integrand, eps, T)

    return surf_data[name]["S0"] + integral


# ==============================================================================
# UNIFIED H AND S DISPATCHERS
# ==============================================================================

def H_species(name, T, nasa_gas, surf_data, cp_poly):
    """
    Return H(T) [kcal/mol] for any species by routing to the correct
    data source:

        Gas-phase species  → NASA polynomial (nasa_H)
        Surface species    → Cp integration  (_H_surf)
        Vacant site        → 0.0             (reference state)

    Parameters
    ----------
    name      : str   — species identifier
    T         : float — temperature in K
    nasa_gas  : dict  — from data_io.read_nasa_data()
    surf_data : dict  — from data_io.read_surface_data()
    cp_poly   : dict  — from fit_surface_cp_polynomials()

    Returns
    -------
    float — H in kcal/mol

    Raises
    ------
    ValueError if `name` is not recognised in any data source.
    """
    if name in nasa_gas:
        return nasa_H(name, T, nasa_gas)

    elif name in surf_data:
        return _H_surf(name, T, surf_data, cp_poly)

    elif name in VACANT_SITES:
        return 0.0

    else:
        raise ValueError(f"H_species: unrecognised species '{name}'")


def S_species(name, T, nasa_gas, surf_data, cp_poly):
    """
    Return S(T) [cal/(mol·K)] for any species by routing to the correct
    data source:

        Gas-phase species  → NASA polynomial (nasa_S)
        Surface species    → Cp/T integration (_S_surf)
        Vacant site        → 0.0              (reference state)

    Parameters
    ----------
    name      : str   — species identifier
    T         : float — temperature in K
    nasa_gas  : dict  — from data_io.read_nasa_data()
    surf_data : dict  — from data_io.read_surface_data()
    cp_poly   : dict  — from fit_surface_cp_polynomials()

    Returns
    -------
    float — S in cal/(mol·K)

    Raises
    ------
    ValueError if `name` is not recognised in any data source.
    """
    if name in nasa_gas:
        return nasa_S(name, T, nasa_gas)

    elif name in surf_data:
        return _S_surf(name, T, surf_data, cp_poly)

    elif name in VACANT_SITES:
        return 0.0

    else:
        raise ValueError(f"S_species: unrecognised species '{name}'")


# ==============================================================================
# ELEMENTARY STEP Keq COMPUTATION
# ==============================================================================

def _delta_rxn(sp_list, T, thermo_fn, nasa_gas, surf_data, cp_poly):
    """
    Compute the reaction change of a thermodynamic quantity for one
    elementary step at temperature T:

        Δ(quantity) = Σᵢ νᵢ · thermo_fn(speciesᵢ, T)

    Parameters
    ----------
    sp_list   : list of (nu, species_name)
    T         : float    — temperature in K
    thermo_fn : callable — either H_species or S_species
    nasa_gas  : dict
    surf_data : dict
    cp_poly   : dict

    Returns
    -------
    float
    """
    return sum(
        nu * thermo_fn(sp, T, nasa_gas, surf_data, cp_poly)
        for nu, sp in sp_list
    )


def compute_step_keq(steps, T_arr, nasa_gas, surf_data, cp_poly):
    """
    Pre-compute Keq(T) for every elementary step over the temperature array.

    For each step and each temperature:
        1. Compute ΔH [kcal/mol] and ΔS [cal/(mol·K)]
        2. Convert ΔH to cal/mol   →  ΔG = ΔH*1000 − T*ΔS   [cal/mol]
        3. Keq = exp(−ΔG / (R·T))

    Parameters
    ----------
    steps     : list[Step]   — elementary steps (see Step dataclass)
    T_arr     : np.ndarray   — temperature array in K
    nasa_gas  : dict         — from data_io.read_nasa_data()
    surf_data : dict         — from data_io.read_surface_data()
    cp_poly   : dict         — from fit_surface_cp_polynomials()

    Returns
    -------
    list of np.ndarray
        One 1-D array per step; array[k] is Keq at T_arr[k].
    """
    step_Keq = []

    for step in steps:
        keq_vals = []

        for T in T_arr:
            dH = _delta_rxn(step.species, T, H_species, nasa_gas, surf_data, cp_poly)
            dS = _delta_rxn(step.species, T, S_species, nasa_gas, surf_data, cp_poly)

            # Convert ΔH from kcal/mol to cal/mol before computing ΔG
            dG = dH * 1000.0 - T * dS   # cal/mol

            keq_vals.append(np.exp(-dG / (R_CAL * T)))

        step_Keq.append(np.array(keq_vals))

    return step_Keq
