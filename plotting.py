"""
plotting.py
===========
All figure-generation functions for the NH3 equilibrium + microkinetics project.

Every function is self-contained: it receives its data as arguments,
draws the figure, saves it to disk, and calls plt.show().  No global
state is read or written.

Public API
----------
    plot_cp_fits(surf_data, cp_poly, nasa_gas, T_CP, save_path)
    plot_elementary_keq(steps, step_Keq, T_arr, save_path)
    plot_overall_keq(T_arr, Keq_overall, Keq_lit, save_path)
    plot_tof(T_arr, tof_arr, save_path)
    plot_surface_coverage(T_arr, ss_array, save_path)
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

from thermodynamics import nasa_Cp
from config import (
    FIG_CP_PATH,
    FIG_KEQSTEP_PATH,
    FIG_KEQOV_PATH,
    FIG_TOF_PATH,
    FIG_COVERAGE_PATH,
    FIG_RATES_PATH,
    FIG_SS_TIME_PATH,
    SDEN_T,
    SDEN_S,
)


# ==============================================================================
# FIGURE 1 — Cp FITS
# ==============================================================================

def plot_cp_fits(
    surf_data,
    cp_poly,
    nasa_gas,
    T_CP,
    save_path=FIG_CP_PATH,
):
    """
    Multi-panel figure showing Cp(T) for every species:

        Surface species  : DFT data points (scatter) + degree-3 fit (line)
        Gas-phase species: NASA polynomial Cp curve only

    Panels are arranged in a 4-column grid; unused panels are hidden.

    Parameters
    ----------
    surf_data : dict       — from data_io.read_surface_data()
    cp_poly   : dict       — from thermodynamics.fit_surface_cp_polynomials()
    nasa_gas  : dict       — from data_io.read_nasa_data()
    T_CP      : np.ndarray — the 15 tabulated Cp temperatures (K)
    save_path : str        — output file path (default from config)
    """
    all_surf  = list(surf_data.keys())
    gas_names = list(nasa_gas.keys())
    n_total   = len(all_surf) + len(gas_names)

    # Dense temperature grid for smooth Cp curves
    T_plot = np.linspace(100, 1500, 400)

    n_cols = 4
    n_rows = int(np.ceil(n_total / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        constrained_layout=True,
    )
    fig.suptitle(
        "Cp(T): DFT data, degree-3 polynomial fit, and NASA polynomial",
        fontsize=14,
        fontweight="bold",
    )

    axes_flat = axes.flatten()

    # ── Surface species ───────────────────────────────────────────────────────
    for idx, name in enumerate(all_surf):
        ax = axes_flat[idx]

        # Scatter: the 15 DFT-derived Cp values at the tabulated temperatures
        ax.scatter(
            T_CP - 273.15, surf_data[name]["Cp"],
            color="black", s=25, zorder=5, label="DFT data",
        )

        # Line: degree-3 polynomial fit on the dense temperature grid
        ax.plot(
            T_plot - 273.15, cp_poly[name](T_plot),
            color="royalblue", lw=2, label="Poly-3 fit",
        )

        ax.set_title(name, fontsize=10)
        ax.set_xlabel("T (°C)", fontsize=8)
        ax.set_ylabel("Cp (cal/mol/K)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ── Gas-phase species (NASA polynomial only — no DFT Cp points) ──────────
    for jdx, gname in enumerate(gas_names):
        ax      = axes_flat[len(all_surf) + jdx]
        cp_nasa = np.array([nasa_Cp(gname, T, nasa_gas) for T in T_plot])

        ax.plot(T_plot - 273.15, cp_nasa, color="crimson", lw=2, label="NASA poly")
        ax.set_title(f"{gname} (gas)", fontsize=10)
        ax.set_xlabel("T (°C)", fontsize=8)
        ax.set_ylabel("Cp (cal/mol/K)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide any leftover empty panels
    for k in range(n_total, len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ==============================================================================
# FIGURE 2 — Keq FOR EACH ELEMENTARY STEP
# ==============================================================================

def plot_elementary_keq(
    steps,
    step_Keq,
    T_arr,
    thermo_label="NASA-7 (Excel)",
    step_Keq_seqsim=None,
    save_path=FIG_KEQSTEP_PATH,
):
    """
    Multi-panel figure of Keq(T) for every elementary step.

    Colour coding:
        Royal blue    — active step (main thermodynamic source)
        Silver        — excluded step (ν = 0)
        Dashed orange — same step from data_ammonia.csv (SeqSim), only when
                        the main source differs (step_Keq_seqsim is not None)

    Parameters
    ----------
    steps            : list[Step]            — from kinetics.STEPS
    step_Keq         : list[np.ndarray]      — from thermodynamics.compute_step_keq()
    T_arr            : np.ndarray            — temperature array in K
    thermo_label     : str                   — label for the main trace (shown in legend)
    step_Keq_seqsim  : list[np.ndarray] or None
                        Keq from SeqSim (data_ammonia.csv), same step order.
                        When provided, a dashed orange trace is overlaid on each panel.
    save_path        : str                   — output file path (default from config)
    """
    n_steps = len(steps)
    n_cols  = 4
    n_rows  = int(np.ceil(n_steps / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        constrained_layout=True,
    )
    fig.suptitle(
        f"Keq(T) for each elementary step  —  {thermo_label}",
        fontsize=14,
        fontweight="bold",
    )

    axes_flat = axes.flatten()

    for idx, step in enumerate(steps):
        ax    = axes_flat[idx]
        color = "royalblue" if step.active else "silver"
        tag   = "[active]" if step.active else "[excluded — ν=0]"

        ax.semilogy(T_arr - 273.15, step_Keq[idx],
                    color=color, lw=2, label=thermo_label)

        if step_Keq_seqsim is not None:
            ax.semilogy(T_arr - 273.15, step_Keq_seqsim[idx],
                        color="darkorange", lw=1.5, linestyle="--",
                        label="SeqSim (CSV)")
            ax.legend(fontsize=6, loc="best")

        ax.set_title(f"{step.label}\n{tag}", fontsize=7.5)
        ax.set_xlabel("T (°C)", fontsize=8)
        ax.set_ylabel("Keq", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, which="both")

    # Hide unused panels
    for k in range(n_steps, len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ==============================================================================
# FIGURE 3 — OVERALL Keq VS LITERATURE
# ==============================================================================

def plot_overall_keq(
    T_arr,
    Keq_overall,
    Keq_lit,
    thermo_label="NASA-7 (Excel)",
    Keq_overall_seqsim=None,
    save_path=FIG_KEQOV_PATH,
):
    """
    Single-panel comparison of the DFT-derived overall Keq(T) against the
    Temkin–Pyzhev literature value, with an optional third trace from
    the SeqSim (data_ammonia.csv) thermodynamics.

    Parameters
    ----------
    T_arr              : np.ndarray      — temperature array in K
    Keq_overall        : np.ndarray      — from kinetics.compute_overall_keq()
    Keq_lit            : np.ndarray      — literature Keq from kinetics.K_lit()
    thermo_label       : str             — label for the main trace
    Keq_overall_seqsim : np.ndarray or None
                          Overall Keq from SeqSim (data_ammonia.csv).
                          When provided, plotted as a dashed orange line.
    save_path          : str             — output file path (default from config)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(
        T_arr - 273.15, Keq_overall,
        color="royalblue", lw=2.5,
        label=f"Overall $K_{{eq}}$ — {thermo_label}",
    )
    ax.semilogy(
        T_arr - 273.15, Keq_lit,
        color="crimson", lw=2.5, linestyle="--",
        label="$K_{lit}$(T)",
    )
    if Keq_overall_seqsim is not None:
        ax.semilogy(
            T_arr - 273.15, Keq_overall_seqsim,
            color="darkorange", lw=2.0, linestyle=":",
            label="SeqSim $K_{eq}$ (CSV)",
        )

    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("$K_{eq}$", fontsize=12)
    ax.set_title(
        "Overall $K_{eq}$: N$_2$ + 3H$_2$ $\\rightleftharpoons$ 2NH$_3$\n"
        "N$_2$(T)+*(T)$\\rightleftharpoons$2N(T) excluded  |  P = 1 bar",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ==============================================================================
# FIGURE 4 — TURNOVER FREQUENCY (TOF) VS TEMPERATURE
# ==============================================================================

def plot_tof(
    T_arr,
    tof_arr,
    t_stop_arr=None,
    ss_reached_arr=None,
    save_path=FIG_TOF_PATH,
):
    """
    Two-panel figure: TOF vs temperature (top) and time-to-steady-state vs
    temperature (bottom).  The bottom panel is only drawn when t_stop_arr is
    provided.

    Parameters
    ----------
    T_arr         : np.ndarray — temperature array [K]
    tof_arr       : np.ndarray — TOF [mol NH3/(mol_site·s)]
    t_stop_arr    : np.ndarray or None — simulated seconds when ODE stopped
    ss_reached_arr: np.ndarray(bool) or None — True where SS event fired
    save_path     : str
    """
    have_timing = t_stop_arr is not None and ss_reached_arr is not None

    if have_timing:
        fig, (ax_tof, ax_t) = plt.subplots(
            2, 1, figsize=(8, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 2]},
        )
    else:
        fig, ax_tof = plt.subplots(figsize=(8, 5))

    T_C = T_arr - 273.15   # convert to °C for all TOF plot axes

    # ── Top panel: TOF ────────────────────────────────────────────────────────
    ax_tof.semilogy(T_C, np.abs(tof_arr), color="royalblue", lw=2.5)
    ax_tof.set_ylabel("TOF  [mol NH$_3$ / (mol$_{site}$ · s)]", fontsize=12)
    ax_tof.set_title(
        "Turnover Frequency — NH$_3$ production\n"
        "TOF = $(r_{net,2} + r_{net,5})$ / SDTOT",
        fontsize=12,
    )
    ax_tof.grid(True, alpha=0.3, which="both")

    if have_timing:
        # Mark temperatures where SS was NOT reached (timed out)
        timeout_mask = ~ss_reached_arr
        if timeout_mask.any():
            ax_tof.semilogy(
                T_C[timeout_mask], np.abs(tof_arr[timeout_mask]),
                "rv", ms=7, zorder=5, label="timeout (not true SS)",
            )
            ax_tof.legend(fontsize=10)

    # ── Bottom panel: time to steady state ───────────────────────────────────
    if have_timing:
        ss_mask = ss_reached_arr
        to_mask = ~ss_reached_arr

        if ss_mask.any():
            ax_t.semilogy(
                T_C[ss_mask], t_stop_arr[ss_mask],
                "o-", color="seagreen", lw=2, ms=4,
                label="SS reached",
            )
        if to_mask.any():
            ax_t.semilogy(
                T_C[to_mask], t_stop_arr[to_mask],
                "rv", ms=7, zorder=5,
                label="timeout",
            )

        ax_t.set_xlabel("Temperature (°C)", fontsize=12)
        ax_t.set_ylabel("$t_{stop}$  [s]", fontsize=12)
        ax_t.set_title("Simulated time to steady state", fontsize=11)
        ax_t.legend(fontsize=10)
        ax_t.grid(True, alpha=0.3, which="both")
    else:
        ax_tof.set_xlabel("Temperature (°C)", fontsize=12)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ==============================================================================
# FIGURE 5 — SURFACE COVERAGES VS TEMPERATURE
# ==============================================================================

def plot_surface_coverage(
    T_arr,
    ss_array,
    save_path=FIG_COVERAGE_PATH,
):
    """
    Three-panel figure of dimensionless surface coverages θ vs temperature.

    Panel layout
    ------------
    Panel 1 (left)   — θ_H:   H(T) terrace  and  H(S) step
    Panel 2 (centre) — θ_N:   N(T) terrace  and  N(S) step
    Panel 3 (right)  — θ_NH3: NH3(T) terrace and NH3(S) step

    A fourth panel shows the vacant-site fractions:
    Panel 4 (far right) — θ_vac: vacant terrace  and  vacant step

    Coverage normalisation
    ----------------------
    Terrace species:
        θ_i(T) = coverage_i [mol/cm²] / SDEN_T [mol/cm²]

    Step species:
        θ_i(S) = coverage_i [mol/cm²] / SDEN_S [mol/cm²]

    Vacant-site coverage is derived from the site balance:
        θ_vac(T) = 1 − Σ θ_i(T)   for all terrace adsorbates
        θ_vac(S) = 1 − Σ θ_i(S)   for all step adsorbates

    Both terrace and step lines are drawn on the same panel with
    different line styles so they are easy to distinguish.

    Parameters
    ----------
    T_arr    : np.ndarray — temperature array [K], shape (n_temps,)
    ss_array : np.ndarray — steady-state results, shape (n_temps, N_VARS).
                            Produced by stacking the ODE final states in
                            main.py.  Column order follows the IDX_* constants
                            defined in microkinetics.py.
    save_path : str       — output file path (default from config)
    """

    # ── Import state-vector index constants from microkinetics ────────────────
    # These are imported here rather than at module top level to avoid a
    # circular import: plotting → microkinetics → (nothing that needs plotting).
    from microkinetics import (
        IDX_HT,   IDX_HS,
        IDX_NT,   IDX_NS,
        IDX_NH3T, IDX_NH3S,
        IDX_N2T,  IDX_N2S,
        IDX_NHT,  IDX_NHS,
        IDX_NH2T, IDX_NH2S,
        IDX_NSL,
    )

    # ── Compute dimensionless coverages [0, 1] ────────────────────────────────
    # Divide each mol/cm² column by the appropriate site density.

    theta_HT   = ss_array[:, IDX_HT]   / SDEN_T
    theta_NT   = ss_array[:, IDX_NT]   / SDEN_T
    theta_NH3T = ss_array[:, IDX_NH3T] / SDEN_T

    theta_HS   = ss_array[:, IDX_HS]   / SDEN_S
    theta_NS   = ss_array[:, IDX_NS]   / SDEN_S
    theta_NH3S = ss_array[:, IDX_NH3S] / SDEN_S

    # ── Vacant-site fractions from the site balance ───────────────────────────
    # Sum all terrace adsorbate coverages and subtract from 1.
    # Every terrace species that appears in the state vector contributes.
    theta_occ_T = (
        ss_array[:, IDX_N2T]  / SDEN_T
        + ss_array[:, IDX_HT]   / SDEN_T
        + ss_array[:, IDX_NH3T] / SDEN_T
        + ss_array[:, IDX_NT]   / SDEN_T
        + ss_array[:, IDX_NHT]  / SDEN_T
        + ss_array[:, IDX_NH2T] / SDEN_T
    )
    theta_vac_T = np.clip(1.0 - theta_occ_T, 0.0, 1.0)

    # Step site balance also includes N(SL) which occupies step-type sites.
    theta_occ_S = (
        ss_array[:, IDX_N2S]  / SDEN_S
        + ss_array[:, IDX_HS]   / SDEN_S
        + ss_array[:, IDX_NH3S] / SDEN_S
        + ss_array[:, IDX_NS]   / SDEN_S
        + ss_array[:, IDX_NHS]  / SDEN_S
        + ss_array[:, IDX_NH2S] / SDEN_S
        + ss_array[:, IDX_NSL]  / SDEN_S
    )
    theta_vac_S = np.clip(1.0 - theta_occ_S, 0.0, 1.0)

    # ── Figure layout: 1 row × 4 panels ──────────────────────────────────────
    fig, axes = plt.subplots(
        1, 4,
        figsize=(20, 5),
        constrained_layout=True,
    )
    fig.suptitle(
        "Steady-state surface coverages vs temperature  (P = 1 bar)",
        fontsize=13,
        fontweight="bold",
    )

    # Shared style helpers — defined once so each panel looks identical
    # except for which data it shows.
    STYLE_TERRACE = dict(lw=2.0, linestyle="-")
    STYLE_STEP    = dict(lw=2.0, linestyle="--")

    T_C = T_arr - 273.15   # convert to °C for all coverage plot axes

    # ── Panel 0 — θ_H ─────────────────────────────────────────────────────────
    ax = axes[0]
    ax.semilogy(T_C, np.clip(theta_HT, 1e-20, None),
                color="steelblue",   label=r"$\theta_H$(T)",   **STYLE_TERRACE)
    ax.semilogy(T_C, np.clip(theta_HS, 1e-20, None),
                color="darkorange",  label=r"$\theta_H$(S)",   **STYLE_STEP)
    ax.set_title(r"$\theta_H$ — Hydrogen coverage", fontsize=11)
    ax.set_xlabel("Temperature (°C)", fontsize=10)
    ax.set_ylabel(r"$\theta$  (dimensionless)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # ── Panel 1 — θ_N ─────────────────────────────────────────────────────────
    ax = axes[1]
    ax.semilogy(T_C, np.clip(theta_NT, 1e-20, None),
                color="steelblue",   label=r"$\theta_N$(T)",   **STYLE_TERRACE)
    ax.semilogy(T_C, np.clip(theta_NS, 1e-20, None),
                color="darkorange",  label=r"$\theta_N$(S)",   **STYLE_STEP)
    ax.set_title(r"$\theta_N$ — Nitrogen coverage", fontsize=11)
    ax.set_xlabel("Temperature (°C)", fontsize=10)
    ax.set_ylabel(r"$\theta$  (dimensionless)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # ── Panel 2 — θ_NH3 ───────────────────────────────────────────────────────
    ax = axes[2]
    ax.semilogy(T_C, np.clip(theta_NH3T, 1e-20, None),
                color="steelblue",   label=r"$\theta_{NH_3}$(T)", **STYLE_TERRACE)
    ax.semilogy(T_C, np.clip(theta_NH3S, 1e-20, None),
                color="darkorange",  label=r"$\theta_{NH_3}$(S)", **STYLE_STEP)
    ax.set_title(r"$\theta_{NH_3}$ — Ammonia coverage", fontsize=11)
    ax.set_xlabel("Temperature (°C)", fontsize=10)
    ax.set_ylabel(r"$\theta$  (dimensionless)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # ── Panel 3 — θ_vac ───────────────────────────────────────────────────────
    ax = axes[3]
    ax.semilogy(T_C, np.clip(theta_vac_T, 1e-20, None),
                color="steelblue",   label=r"$\theta_{vac}$(T)", **STYLE_TERRACE)
    ax.semilogy(T_C, np.clip(theta_vac_S, 1e-20, None),
                color="darkorange",  label=r"$\theta_{vac}$(S)", **STYLE_STEP)
    ax.set_title(r"$\theta_{vac}$ — Vacant sites", fontsize=11)
    ax.set_xlabel("Temperature (°C)", fontsize=10)
    ax.set_ylabel(r"$\theta$  (dimensionless)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ==============================================================================
# FIGURE 5b — TIME TO STEADY STATE VS TEMPERATURE
# ==============================================================================

def plot_ss_time(
    T_arr,
    t_stop_arr,
    ss_reached_arr,
    save_path=FIG_SS_TIME_PATH,
):
    """
    Single-panel semilog plot of the simulated time at which the ODE stopped
    for each temperature, colour-coded by whether the steady-state event fired
    (green circles) or the integration hit the maximum time limit (red triangles).

    Parameters
    ----------
    T_arr          : np.ndarray      — temperature array [K]
    t_stop_arr     : np.ndarray      — t at which solve_ivp stopped [s]
    ss_reached_arr : np.ndarray(bool)— True where SS event fired
    save_path      : str
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ss_mask = ss_reached_arr
    to_mask = ~ss_reached_arr

    T_C = T_arr - 273.15

    if ss_mask.any():
        ax.semilogy(
            T_C[ss_mask], t_stop_arr[ss_mask],
            "o-", color="seagreen", lw=2, ms=5,
            label="Steady state reached",
        )
    if to_mask.any():
        ax.semilogy(
            T_C[to_mask], t_stop_arr[to_mask],
            "rv", ms=8, zorder=5,
            label="Timeout — not true SS",
        )

    n_ss  = ss_mask.sum()
    n_to  = to_mask.sum()
    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("Simulated time $t_{stop}$  [s]", fontsize=12)
    ax.set_title(
        f"Time to steady state per temperature point\n"
        f"SS event: {n_ss}/{len(T_arr)} points  |  "
        f"Timeout: {n_to}/{len(T_arr)} points",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ==============================================================================
# FIGURE 6 — FORWARD, BACKWARD, AND NET REACTION RATES VS TEMPERATURE
# ==============================================================================

def plot_reaction_rates(
    T_arr,
    rf_matrix,
    rb_matrix,
    rnet_matrix,
    steps,
    save_path=FIG_RATES_PATH,
):
    """
    Multi-panel figure showing rf(T), rb(T), and rnet(T) for all 22
    elementary steps, evaluated at the steady-state surface coverages
    and gas-phase concentrations at each temperature.

    All three rates share the same subplot so their relative magnitudes
    are directly readable.  A semilog-y axis is used because the rates
    span many orders of magnitude.

    Sign convention for rnet
    ------------------------
    rnet = rf - rb.
    Positive rnet → net forward direction (left-to-right as written).
    Negative rnet → net reverse direction.
    Because semilogy cannot plot negative values, |rnet| is plotted and
    the subplot title flags the direction with "(net fwd.)" or "(net rev.)".

    Panel layout
    ------------
    22 steps arranged in a 4-column grid (6 rows × 4 columns).
    The last two panels in row 6 are hidden (22 is not divisible by 4).

    Colour coding
    -------------
    Blue solid  — rf        (forward rate)
    Red  solid  — rb        (backward rate)
    Black solid — |rnet|    net forward  (rnet ≥ 0 on average)
    Black dash  — |rnet|    net reverse  (rnet < 0 on average)

    Parameters
    ----------
    T_arr       : np.ndarray — temperature array [K], shape (n_temps,)
    rf_matrix   : np.ndarray — forward  rates, shape (n_temps, n_steps)
                               rf_matrix[i, j] = rf of step j at T_arr[i]
                               units: mol/(cm²·s)
    rb_matrix   : np.ndarray — backward rates, shape (n_temps, n_steps)
    rnet_matrix : np.ndarray — net      rates, shape (n_temps, n_steps)
    steps       : list[Step] — from kinetics.STEPS; used for subplot titles
    save_path   : str        — output file path (default from config)
    """
    n_steps = len(steps)
    n_cols  = 4
    n_rows  = int(np.ceil(n_steps / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        constrained_layout=True,
    )
    fig.suptitle(
        "Steady-state reaction rates vs temperature\n"
        r"$r_f$ (blue), $r_b$ (red), $|r_{net}|$ (black)"
        "  —  solid: net forward, dashed: net reverse",
        fontsize=13,
        fontweight="bold",
    )

    axes_flat = axes.flatten()

    T_C = T_arr - 273.15   # convert to °C for all reaction-rate plot axes

    # Small floor applied before semilogy to guard against exact zeros.
    # 1e-60 is far below any physically meaningful rate and prevents
    # log(0) errors without distorting the visible part of the plot.
    RATE_FLOOR = 1.0e-60

    for j, step in enumerate(steps):
        ax = axes_flat[j]

        # Extract the three rate arrays for this step across all temperatures
        rf_j    = rf_matrix[:, j]
        rb_j    = rb_matrix[:, j]
        rnet_j  = rnet_matrix[:, j]

        # Determine whether rnet is predominantly forward or reverse.
        # We use the mean across the temperature sweep as the indicator.
        # This handles the common case where a step is quasi-equilibrated
        # (rnet ≈ 0) or clearly one-directional.
        net_is_reverse = np.mean(rnet_j) < 0.0

        rnet_linestyle = "--" if net_is_reverse else "-"
        direction_tag  = "(net rev.)" if net_is_reverse else "(net fwd.)"

        # ── Forward rate ──────────────────────────────────────────────────────
        ax.semilogy(
            T_C,
            np.clip(rf_j, RATE_FLOOR, None),
            color     = "steelblue",
            lw        = 1.8,
            linestyle = "-",
            label     = r"$r_f$",
        )

        # ── Backward rate ─────────────────────────────────────────────────────
        ax.semilogy(
            T_C,
            np.clip(rb_j, RATE_FLOOR, None),
            color     = "crimson",
            lw        = 1.8,
            linestyle = "-",
            label     = r"$r_b$",
        )

        # ── Absolute net rate — dashed when net direction is reverse ──────────
        # np.abs is applied so semilogy never receives a negative value.
        # The direction is communicated by linestyle and the title tag instead.
        ax.semilogy(
            T_C,
            np.clip(np.abs(rnet_j), RATE_FLOOR, None),
            color     = "black",
            lw        = 2.0,
            linestyle = rnet_linestyle,
            label     = r"$|r_{net}|$",
        )

        # Subplot title: reaction label on line 1,
        # active/excluded status and net direction on line 2
        active_tag = "active" if step.active else "excluded"
        ax.set_title(
            f"{step.label}\n[{active_tag}]  {direction_tag}",
            fontsize=7.0,
        )

        ax.set_xlabel("T (°C)", fontsize=8)
        ax.set_ylabel("Rate  [mol/(cm²·s)]", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="best")
        ax.grid(True, alpha=0.3, which="both")

    # Hide unused panels in the last row (22 steps, 24 panels → 2 unused)
    for k in range(n_steps, len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
