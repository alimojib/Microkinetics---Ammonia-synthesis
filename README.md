# Microkinetics — Ammonia Synthesis

A modular Python implementation of a **power-law microkinetic model** for ammonia synthesis over a stepped ruthenium (or iron) catalyst surface. The model integrates DFT-derived thermochemistry with Hertz–Knudsen adsorption kinetics and Brønsted–Evans–Polanyi (BEP) surface-reaction barriers to predict steady-state turnover frequencies (TOF), surface coverages, and elementary-step rates across the industrially relevant 573–873 K window at 50 bar.

---

## Table of Contents

1. [Overview](#overview)
2. [Physical & Chemical System](#physical--chemical-system)
3. [Reaction Network](#reaction-network)
   - [Elementary Steps](#elementary-steps)
   - [Active vs Inactive Steps](#active-vs-inactive-steps)
4. [Governing Equations](#governing-equations)
   - [Thermodynamics](#thermodynamics)
   - [Equilibrium Constants](#equilibrium-constants)
   - [Rate Constants](#rate-constants)
   - [CSTR ODE System](#cstr-ode-system)
5. [Numerical Methods](#numerical-methods)
6. [Module Architecture](#module-architecture)
7. [Output Figures](#output-figures)
8. [Input Data](#input-data)
9. [Configuration Reference](#configuration-reference)
10. [Installation & Usage](#installation--usage)
11. [State Vector Reference](#state-vector-reference)
12. [Key Design Decisions](#key-design-decisions)

---

## Overview

The model answers the question: *given DFT-derived energetics for every elementary step in the NH₃ synthesis mechanism, what steady-state NH₃ production rate does a CSTR reactor produce as a function of temperature?*

The workflow is:

```
Excel data (DFT Cp, H, S + NASA polynomials + BEP params)
        ↓
Fit surface Cp polynomials          → thermodynamics module
        ↓
Compute Keq(T) per elementary step  → thermodynamics + kinetics modules
        ↓
Verify overall Keq vs literature    → kinetics module + plotting
        ↓
Precompute kf / kb matrices         → microkinetics module
        ↓
Integrate CSTR ODE to steady state  → microkinetics module
        ↓
Extract TOF, coverages, rates       → microkinetics + plotting modules
```

---

## Physical & Chemical System

| Property | Value |
|---|---|
| Reactor type | CSTR (Continuous Stirred Tank Reactor) |
| Reactor volume | 7.7 cm³ |
| Total pressure | 50 bar |
| Temperature sweep | 573–873 K (5 K steps) |
| N₂ feed flow | 11.4 sccm |
| H₂ feed flow | 34.2 sccm |
| H₂ : N₂ ratio | 3 : 1 (stoichiometric) |
| NH₃ in feed | 0 (pure reactant feed) |
| Catalyst surface area / reactor volume (ABYV) | 1200 cm²/cm³ |
| Total site density (SDTOT) | 2.6188 × 10⁻⁹ mol/cm² |
| Step-site fraction | 2 % of total sites |
| Sticking coefficient | 0.5 (all adsorbing species) |

The **overall reaction** is:

$$\text{N}_2 + 3\,\text{H}_2 \;\rightleftharpoons\; 2\,\text{NH}_3 \qquad \Delta H_{298} \approx -92 \;\text{kJ mol}^{-1}$$

The catalyst surface is modelled with **two distinct site types**:

| Symbol | Description |
|---|---|
| `*(T)` | Terrace site — the majority surface (98 % of total) |
| `*(S)` | Step site — the minority but highly active site (2 % of total); primary N₂ dissociation locus |
| `*(SL)` | Lower-step site — N landing site for the step dissociation channel N₂(S) + *(SL) → N(S) + N(SL) |

---

## Reaction Network

### Elementary Steps

The full network consists of **22 reversible elementary steps**:

| Index | Reaction | Type | Active |
|---|---|---|---|
| 0 | N₂ + \*(T) ⇌ N₂(T) | Terrace adsorption | ✓ |
| 1 | H₂ + 2\*(T) ⇌ 2H(T) | Terrace adsorption (dissociative) | ✓ |
| 2 | NH₃ + \*(T) ⇌ NH₃(T) | Terrace adsorption | ✓ |
| 3 | N₂ + \*(S) ⇌ N₂(S) | Step adsorption | ✓ |
| 4 | H₂ + 2\*(S) ⇌ 2H(S) | Step adsorption (dissociative) | ✓ |
| 5 | NH₃ + \*(S) ⇌ NH₃(S) | Step adsorption | ✓ |
| 6 | N(T) + \*(S) ⇌ N(S) + \*(T) | Terrace→step N transfer | ✗ |
| 7 | NH(T) + \*(S) ⇌ NH(S) + \*(T) | Terrace→step NH transfer | ✗ |
| 8 | NH₂(T) + \*(S) ⇌ NH₂(S) + \*(T) | Terrace→step NH₂ transfer | ✗ |
| 9 | NH₃(T) + \*(S) ⇌ NH₃(S) + \*(T) | Terrace→step NH₃ transfer | ✗ |
| 10 | H(T) + \*(S) ⇌ H(S) + \*(T) | Terrace→step H transfer | ✗ |
| 11 | N₂(T) + \*(T) ⇌ 2N(T) | Terrace N₂ dissociation | ✓ |
| 12 | N₂(S) + \*(SL) ⇌ N(S) + N(SL) | Step N₂ dissociation (to lower step) | ✓ |
| 13 | N₂(S) + \*(S) ⇌ 2N(S) | On-step N₂ dissociation | ✓ |
| 14 | N₂(S) + \*(T) ⇌ N(S) + N(T) | Cross-site N₂ dissociation | ✗ |
| 15 | N(T) + \*(SL) ⇌ N(SL) + \*(T) | N terrace→lower-step transfer | ✗ |
| 16 | N(T) + H(T) ⇌ NH(T) + \*(T) | Terrace hydrogenation step 1 | ✓ |
| 17 | NH(T) + H(T) ⇌ NH₂(T) + \*(T) | Terrace hydrogenation step 2 | ✓ |
| 18 | NH₂(T) + H(T) ⇌ NH₃(T) + \*(T) | Terrace hydrogenation step 3 | ✓ |
| 19 | N(S) + H(S) ⇌ NH(S) + \*(S) | Step hydrogenation step 1 | ✓ |
| 20 | NH(S) + H(S) ⇌ NH₂(S) + \*(S) | Step hydrogenation step 2 | ✓ |
| 21 | NH₂(S) + H(S) ⇌ NH₃(S) + \*(S) | Step hydrogenation step 3 | ✓ |

### Active vs Inactive Steps

**Active steps** (✓, 15 total) enter the linear-combination solve that verifies the overall equilibrium constant:

$$K_{\text{eq,overall}}(T) = \prod_{i \in \text{active}} K_{\text{eq},i}(T)^{\nu_i}$$

where the multipliers νᵢ are found by solving A·ν = b with b encoding N₂ + 3H₂ → 2NH₃.

**Inactive steps** (✗, 7 total) are terrace↔step diffusion/transfer steps. They are included in the ODE system structure but their pre-exponential factor A = 0, so kf = kb = 0. They are excluded from the Keq consistency check (νᵢ = 0).

---

## Governing Equations

### Thermodynamics

**Gas-phase species** (N₂, H₂, NH₃) use **NASA 7-coefficient polynomials** (piecewise, two temperature ranges):

$$\frac{C_p}{R} = a_1 + a_2 T + a_3 T^2 + a_4 T^3 + a_5 T^4$$

$$\frac{H}{RT} = a_1 + \frac{a_2 T}{2} + \frac{a_3 T^2}{3} + \frac{a_4 T^3}{4} + \frac{a_5 T^4}{5} + \frac{a_6}{T}$$

$$\frac{S}{R} = a_1 \ln T + a_2 T + \frac{a_3 T^2}{2} + \frac{a_4 T^3}{3} + \frac{a_5 T^4}{4} + a_7$$

**Surface species** use **DFT-derived reference enthalpies and entropies at 0 K** (H₀, S₀) plus a degree-3 polynomial fit to discrete Cp(T) values at 15 tabulated temperatures (100–1500 K):

$$H(T) = H_0 + \int_0^T C_p(T')\,dT'$$

$$S(T) = S_0 + \int_0^T \frac{C_p(T')}{T'}\,dT'$$

The integrals are evaluated numerically with `scipy.integrate.quad`.

**Vacant sites** (\*(T), \*(S), \*(SL)) carry zero enthalpy and entropy by convention (reference state).

### Equilibrium Constants

For each elementary step at temperature T:

$$\Delta G_i(T) = \Delta H_i(T) - T\,\Delta S_i(T)$$

$$K_{\text{eq},i}(T) = \exp\!\left(-\frac{\Delta G_i(T)}{R\,T}\right)$$

where ΔH and ΔS are summed over all species in the step with their stoichiometric coefficients.

The **literature reference** (Temkin–Pyzhev) for the overall equilibrium constant is:

$$\log_{10} K_{\text{lit}} = 2 \left[ 2.1 + \frac{1}{4.571}\!\left(\frac{9591}{T} - 4.6\times10^{-4}\,T + 8.5\times10^{-7}\,T^2\right) - \frac{4.98}{1.985}\log_{10} T \right]$$

(squared to match the 2NH₃ stoichiometry convention used throughout).

### Rate Constants

**Adsorption steps (0–5) — Hertz–Knudsen:**

$$k_f^{\text{ads}} = \frac{S_0}{\Gamma_\text{tot}} \sqrt{\frac{R_\text{SI}\,T}{2\pi\,M_w}} \exp\!\left(-\frac{E_a}{R_\text{cal}\,T}\right)$$

where:
- S₀ = 0.5 (sticking coefficient)
- Γ_tot = 2.6188 × 10⁻⁹ mol/cm² (total site density)
- M_w = molecular weight in kg/mol
- Eₐ = 0 kcal/mol (no adsorption barrier assumed)

$$k_b^{\text{ads}} = \frac{k_f^{\text{ads}}}{K_c}, \qquad K_c = K_p \cdot \frac{R\,T}{P^\circ} \quad \left[\frac{\text{cm}^3}{\text{mol}}\right]$$

where $P^\circ = 1\,\text{bar}$ is the thermodynamic standard state and $R = 83.14\,\text{cm}^3\!\cdot\!\text{bar}/(\text{mol}\!\cdot\!\text{K})$.  The factor $R T / P^\circ \approx 49{,}300\,\text{cm}^3/\text{mol}$ at 593 K converts the dimensionless pressure-based $K_p$ into the concentration-based $K_c$ required by the mixed gas/surface rate expressions (see [Key Design Decisions](#key-design-decisions)).

**Surface reaction steps (11–13, 16–21) — Arrhenius + BEP:**

The activation energy is temperature-dependent through the reaction enthalpy:

$$E_a(T) = \alpha \cdot \Delta H_{\text{rxn}}(T) + E_0 \qquad [\text{kcal/mol}]$$

(floored at 0; negative barriers are unphysical in the Arrhenius framework)

$$k_f^{\text{surf}} = \frac{A \cdot T^\beta}{\text{ABYV}} \exp\!\left(-\frac{E_a(T)}{R_\text{cal}\,T}\right)$$

where β = 1 (modified Arrhenius) and A = α₁ from the Excel kinetic parameters table.

**Diffusion/transfer steps (6–10, 14–15):** A = 0 → kf = kb = 0 (inactive at this stage).

### CSTR ODE System

The model is formulated as a **CSTR** (isothermal, isobaric, well-mixed). There are **16 coupled ODEs** — 13 surface coverage equations and 3 gas-phase concentration equations.

**Surface species** (not convected — remain on catalyst):

$$\frac{d\theta_i}{dt} = \sum_j \nu_{ij}\, r_{\text{net},j} \qquad [\text{mol/(cm}^2 \cdot \text{s)}]$$

**Gas-phase species** (CSTR balance with inlet and outlet flows):

$$\frac{dC_i}{dt} = \sum_j \nu_{ij}\, r_{\text{net},j} \cdot \text{ABYV} + \frac{Q}{V}\left(C_{i,\text{feed}} - C_i\right) \qquad [\text{mol/(cm}^3 \cdot \text{s)}]$$

where:
- ABYV = 1200 cm²/cm³ (area-to-volume ratio, converts surface rate → volumetric)
- Q/V = (Q_in / V_reactor) = 0.76/7.7 ≈ 0.0987 s⁻¹ (reciprocal residence time)
- C_i,feed = (xᵢ · P_bar) / (R_cm³·bar · T) from the ideal gas law

**Site balances** constrain vacant-site concentrations:

$$\theta_{\text{vac},T} = \Gamma_T - \theta_{N_2(T)} - \theta_{N(T)} - \theta_{H(T)} - \theta_{NH_3(T)} - \theta_{NH_2(T)} - \theta_{NH(T)}$$

$$\theta_{\text{vac},S} = \Gamma_S - \theta_{N_2(S)} - \theta_{N(S)} - \theta_{H(S)} - \theta_{NH_3(S)} - \theta_{NH_2(S)} - \theta_{NH(S)} - \theta_{N(SL)}$$

**Turnover frequency:**

$$\text{TOF} = \left(-r_{\text{net},2} - r_{\text{net},5}\right) \cdot \frac{\text{ABYV}}{\Gamma_\text{tot}} \qquad [\text{mol NH}_3 \cdot \text{mol}_\text{site}^{-1} \cdot \text{s}^{-1}]$$

where steps 2 and 5 are the NH₃(T) and NH₃(S) desorption steps (negative r_net = net desorption = NH₃ production).

---

## Numerical Methods

| Aspect | Choice | Rationale |
|---|---|---|
| ODE solver | `scipy.integrate.solve_ivp`, method `Radau` | Implicit RK5; designed for stiff systems; stiffness ratio can exceed 10¹⁰ in microkinetics |
| Relative tolerance | 10⁻⁸ | Balances accuracy and speed |
| Absolute tolerance | 10⁻³⁰ mol/cm² (surface), 10⁻¹⁸ mol/cm³ (gas) | Per-variable vector; necessary because surface and gas concentrations differ by ~10 orders of magnitude at 50 bar |
| Steady-state detection | Event: max\|dy/dt\| < 10⁻²⁵ | Stops integration early once SS is reached; avoids integrating 10⁶ s past convergence |
| Warm-start strategy | Inherit surface coverages from T_prev; recalculate gas from ideal gas law at T_next | Eliminates violent adsorption transient at each new temperature; reduces solver stiffness by ~10⁷ |
| First-temperature IC | Langmuir pre-equilibration on terrace and step sites | Places surface near adsorption/desorption balance at t=0; prevents step-size collapse at 50 bar |
| Vacant-site floor | EPS_VAC = 10⁻³⁰ mol/cm² | Replaces hard clip-to-zero; eliminates kink in ODE RHS that breaks the Jacobian estimate |
| Rate-constant precomputation | kf/kb matrices (n_temps × n_steps) built once before the ODE loop | Avoids repeating `quad` integrations and BEP calculations inside the hot ODE path |

---

## Module Architecture

```
main.py              — Entry point; orchestrates all steps in order
config.py            — All constants, paths, tolerances (single source of truth)
data_io.py           — Excel reader (3 sheets); returns plain dicts, no calculation
thermodynamics.py    — NASA polynomials + surface Cp fits + H/S/Keq per step
kinetics.py          — Step definitions, stoichiometry matrix, overall Keq solver
microkinetics.py     — Rate constants, ODE RHS, initial conditions, ODE runner
plotting.py          — All 6 figures; each function is self-contained
```

### Data flow

```
config.py ─────────────────────────────────────────────────┐
                                                            │
data_io.py ──► surf_data, nasa_gas, kin_params             │
                    │                                       │
thermodynamics.py ◄─┘  ──► cp_poly, step_Keq               │
                                  │                        │
kinetics.py ◄─────────────────────┘  ──► STEPS, nu_vec     │
                                              │             │
microkinetics.py ◄────────────────────────────┘ ◄──────────┘
                       ──► kf_matrix, kb_matrix, ss_array, tof_arr
                                              │
plotting.py ◄──────────────────────────────────
```

---

## Output Figures

| File | Description |
|---|---|
| `cp_fits.png` | Multi-panel: Cp(T) for all surface species (DFT points + poly-3 fit) and gas-phase species (NASA polynomial) |
| `keq_elementary.png` | Multi-panel: Keq(T) on a log scale for each of the 22 elementary steps; active steps in blue, excluded in silver |
| `keq_overall.png` | DFT-derived overall Keq(T) vs Temkin–Pyzhev literature reference; validates thermodynamic consistency |
| `tof.png` | NH₃ turnover frequency [mol NH₃ mol_site⁻¹ s⁻¹] vs temperature on a semilog scale |
| `surface_coverage.png` | Four-panel: θ_H, θ_N, θ_NH₃, and θ_vac for terrace (solid) and step (dashed) sites vs temperature |
| `reaction_rates.png` | Multi-panel: forward (blue), backward (red), and |net| (black) rates vs temperature for all 22 steps |

---

## Input Data

All thermodynamic and kinetic parameters are read from a single Excel file. The path is set in `config.py`:

```python
EXCEL_PATH = r"path/to/sciadv.abl6576_dataset_file.xlsx"
```

The file contains three sheets:

| Sheet | Contents |
|---|---|
| `Thermodynamic Properties` | DFT-derived H(0 K) [kcal/mol], S(0 K) [cal/mol/K], and Cp at 15 temperatures for all surface species |
| `NASA Polynomials-0% Strain` | 7-coefficient NASA polynomials (low and high T ranges) for NH₃, N₂, H₂ |
| `Kinetic Parameters` | BEP parameters (α, E₀) from Table 6 and pre-exponential factors (α₁) from Table 8 for all 9 active surface-reaction steps |

The source dataset is from:
> *Science Advances* **abl6576** supplementary dataset.

---

## Configuration Reference

All tunable parameters live in [`config.py`](config.py). The most commonly adjusted ones are:

| Parameter | Default | Description |
|---|---|---|
| `EXCEL_PATH` | — | Path to the input Excel file — **update this first** |
| `P_BAR` | `50.0` | Total pressure [bar] |
| `T_ARR` | `np.arange(573, 874, 5)` | Temperature sweep [K] |
| `RATIO_S` | `0.02` | Step-site fraction of total sites |
| `ABYV` | `1200.0` | Catalyst area / reactor volume [cm²/cm³] |
| `STICKING_COEFF` | `0.5` | Hertz–Knudsen sticking coefficient |
| `ODE_RTOL` | `1e-8` | ODE relative tolerance |
| `ODE_ATOL_SURF` | `1e-30` | ODE absolute tolerance — surface species |
| `ODE_ATOL_GAS` | `1e-18` | ODE absolute tolerance — gas species |
| `SS_TOL` | `1e-25` | Steady-state detection threshold [mol/(cm²·s)] |
| `CP_POLY_DEGREE` | `3` | Degree of polynomial fit to surface Cp(T) data |
| `Q_H2_SCCM` | `34.2` | H₂ feed flow [sccm] |
| `Q_N2_SCCM` | `11.4` | N₂ feed flow [sccm] |

---

## Installation & Usage

### Requirements

```
numpy
scipy
matplotlib
pandas
openpyxl
```

Install with:

```bash
pip install numpy scipy matplotlib pandas openpyxl
```

### Running the model

1. **Set the Excel path** in `config.py`:

   ```python
   EXCEL_PATH = r"path/to/sciadv.abl6576_dataset_file.xlsx"
   ```

2. **Run**:

   ```bash
   python main.py
   ```

The console prints:
- Linear-combination multipliers νᵢ for each step
- Residual of the stoichiometry check (should be ≈ 10⁻¹⁴)
- Net stoichiometry of the combined reaction
- Per-temperature ODE progress: T, N₂ conversion, NH₃ concentration, integration time, status (SS / OK / FAILED)
- Steady-state surface coverages at T ≈ 700 K
- Rate matrix dimensions

Six PNG figures are written to the working directory.

---

## State Vector Reference

The ODE state vector `y` has **16 entries** (indexed 0–15):

| Index | Name | Units | Site |
|---|---|---|---|
| 0 | N₂(T) | mol/cm² | Terrace |
| 1 | H(T) | mol/cm² | Terrace |
| 2 | NH₃(T) | mol/cm² | Terrace |
| 3 | N(T) | mol/cm² | Terrace |
| 4 | NH(T) | mol/cm² | Terrace |
| 5 | NH₂(T) | mol/cm² | Terrace |
| 6 | N₂(S) | mol/cm² | Step |
| 7 | H(S) | mol/cm² | Step |
| 8 | NH₃(S) | mol/cm² | Step |
| 9 | N(S) | mol/cm² | Step |
| 10 | NH(S) | mol/cm² | Step |
| 11 | NH₂(S) | mol/cm² | Step |
| 12 | N(SL) | mol/cm² | Lower-step |
| 13 | N₂ (gas) | mol/cm³ | Gas phase |
| 14 | H₂ (gas) | mol/cm³ | Gas phase |
| 15 | NH₃ (gas) | mol/cm³ | Gas phase |

---

## Key Design Decisions

**Why CSTR instead of batch?**
A batch model depletes reactants to zero, causing the gas-phase concentrations to crash and the ODE to stall at very long times. The CSTR formulation keeps gas-phase concentrations finite at steady state (balanced by inlet flow), which is both physically realistic and numerically much better conditioned.

**Why Radau instead of LSODA?**
The stiffness ratio of the microkinetic ODE can exceed 10¹⁰–10¹⁵. LSODA can oscillate between its Adams and BDF sub-solvers on such problems, resulting in thousands of rejected steps. Radau is a fully implicit method that handles extreme stiffness reliably at the cost of more work per step.

**Why precompute kf/kb matrices?**
Rate constants depend only on temperature, not on ODE state. Precomputing them once before the temperature sweep avoids repeating `scipy.integrate.quad` calls and BEP calculations on every single ODE function evaluation. Inside a stiff ODE with thousands of internal steps per temperature point, this is a significant saving.

**Why the Langmuir warm-start?**
Starting from a clean surface at 50 bar causes a violent adsorption transient: all six Hertz–Knudsen kf values are large and gas-phase concentrations are 50× higher than at 1 bar. The resulting spike in dy/dt forces step sizes of ~10⁻²⁰ s, collapsing the integration. A Langmuir pre-equilibration estimate places the surface near the adsorption/desorption balance from t = 0.

**Why a smooth vacant-site floor (EPS_VAC) instead of hard clipping?**
A hard `max(vac, 0)` creates a kink (non-differentiability) in the ODE right-hand side at exactly zero. Implicit solvers estimate the Jacobian by finite differences; a kink makes that estimate unreliable, causing the step-size controller to collapse to machine epsilon. EPS_VAC = 10⁻³⁰ mol/cm² is physically indistinguishable from zero but keeps the function smooth.

**Why log-residuals for the Keq linear-combination check?**
Not used here — the check is a simple linear algebra solve (A·ν = b) that directly verifies whether the elementary steps combine to give N₂ + 3H₂ → 2NH₃. The residual max|Aν − b| should be ≈ 10⁻¹⁴ (machine precision) for a thermodynamically consistent dataset.

**Why does kb for adsorption steps use Kc instead of Kp?**
`compute_step_keq` returns $K_p$ — the dimensionless thermodynamic equilibrium constant expressed in terms of activities ($P/P^\circ$ for gas species, fractional coverage $\theta$ for surface species, standard state $P^\circ = 1\,\text{bar}$).

For **pure surface steps** (e.g. N₂(T) + \*(T) ⇌ 2N(T)) every species lives in the same surface units (mol/cm²), so the concentration-based $K_c$ is also dimensionless and equals $K_p$ — no correction is needed.

For **adsorption steps** (e.g. N₂ + \*(T) ⇌ N₂(T)) the rate expression mixes a gas concentration $[\text{N}_2]$ in mol/cm³ with a surface concentration $[\text{vac\_T}]$ in mol/cm².  Writing out the concentration-based equilibrium constant:

$$K_c = \frac{[\text{N}_2(\text{T})]\,[\text{mol/cm}^2]}{[\text{N}_2]\,[\text{mol/cm}^3]\;\cdot\;[\text{vac\_T}]\,[\text{mol/cm}^2]} = \frac{\text{cm}^3}{\text{mol}}$$

Because $K_p$ is dimensionless but $K_c$ carries units of cm³/mol, they are related through the molar volume of an ideal gas at standard state:

$$K_c = K_p \cdot \frac{R\,T}{P^\circ}$$

Omitting this conversion makes $k_b = k_f / K_p$ too large by a factor $R T / P^\circ \approx 49{,}300\,\text{cm}^3/\text{mol}$ at 593 K.  In practice:
- Molecular adsorption (N₂, NH₃): surface coverage of the adsorbed intermediate is underestimated by $\approx 49{,}000\times$.
- Dissociative adsorption (H₂ → 2H\*): because the reverse rate involves H\*², the coverage error is $\sqrt{49{,}000} \approx 220\times$.
- With three consecutive hydrogenation steps each requiring H(T), the TOF error compounds to $\approx 220^3 \approx 10^7$, explaining the ~6–7 order of magnitude TOF discrepancy observed before this fix.
