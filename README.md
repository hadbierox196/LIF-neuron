#  Leaky Integrate-and-Fire (LIF) Neuron Model

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-013243.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.3%2B-11557c.svg)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A simulation of the **Leaky Integrate-and-Fire (LIF) neuron** — the simplest widely-used spiking neuron model, capturing how a neuron charges up, leaks, and fires without modeling the biophysical machinery of ion channels.

---

## What it does

- Simulates a noisy LIF neuron via Euler integration with a refractory period
- Plots voltage traces under sub-threshold, supra-threshold, and strong current drive
- Measures the membrane time constant (τ_m) directly from simulated decay/rise curves and compares to theory
- Builds an **f-I curve** (firing rate vs. injected current) and compares simulation against the exact analytical solution
- Computes inter-spike interval (ISI) statistics, including the coefficient of variation as a measure of spike-timing irregularity

---

## Background

Unlike the biophysically detailed Hodgkin-Huxley model, the LIF neuron treats the membrane as a simple **leaky capacitor**: current flows in, voltage rises, and it "leaks" back toward rest. When voltage crosses a threshold, the neuron fires a spike and resets — with no explicit spike-generating currents.

| Concept | What it captures |
|---|---|
| **Rheobase (I_rheo)** | Minimum constant current needed to ever fire |
| **Membrane time constant (τ_m)** | How fast voltage responds to input — sets the "leak" speed |
| **Refractory period (t_ref)** | Enforces a maximum possible firing rate |
| **ISI coefficient of variation** | Spike-timing regularity — 0 for noiseless/regular firing, ~1 for Poisson-like biological spiking |

Despite its simplicity, the LIF model reproduces key neural input-output properties (like the f-I curve) almost exactly, which is why it's the workhorse neuron model in large-scale spiking network simulations.

---

## Installation

```bash
pip install numpy matplotlib
```

## Usage

```bash
python lif_neuron.py
```

Runs all three simulations, saves 3 figures to the working directory, and prints summary spike statistics.

### Core function

```python
V, spike_times = simulate_lif(
    I_ext, t, dt, tau_m, V_rest, V_thresh, V_reset, R_m, t_ref,
    sigma=1.0   # noise strength (mV); set to 0 for deterministic dynamics
)
```

---

## Outputs

| Figure | Shows |
|---|---|
| `fig1_voltage_traces.png` | Voltage traces at 0.8×, 1.5×, and 3.0× rheobase current, with spike counts and firing rates |
| `fig2_subthreshold.png` | (A) Exponential decay after a brief pulse, used to measure τ_m; (B) exponential rise to steady state under a sub-threshold step current |
| `fig3_fi_curve.png` | Simulated firing rate vs. current, overlaid on the exact analytical F-I curve |

---

## Model parameters (defaults)

| Parameter | Value | Meaning |
|---|---|---|
| `tau_m` | 20 ms | Membrane time constant |
| `V_rest` | −70 mV | Resting potential |
| `V_thresh` | −55 mV | Spike threshold |
| `V_reset` | −75 mV | Post-spike reset potential |
| `R_m` | 10 MΩ | Membrane resistance |
| `t_ref` | 2 ms | Refractory period |
| `sigma` | 1.0 mV | Noise strength |

---

## Math, briefly

**Subthreshold dynamics:** `dV/dt = [−(V − V_rest) + R_m·I_ext] / τ_m` (+ noise)

**Spike rule:** if `V ≥ V_thresh` → record spike, set `V = V_reset`, hold for `t_ref`

**Rheobase:** `I_rheo = (V_thresh − V_rest) / R_m`

**Analytical F-I curve (noiseless):**
```
V_ss = V_rest + R_m·I
T_ISI = τ_m·ln[(V_ss − V_reset)/(V_ss − V_thresh)] + t_ref     (if V_ss > V_thresh)
f = 1000 / T_ISI  (Hz)
```

**Max firing rate:** `f_max = 1000 / t_ref`

---

## Roadmap

- Adaptive threshold / adaptation currents (AdEx-style extensions)
- Synaptic input (conductance-based) instead of pure current injection
- Network-level simulations with LIF populations
- Alternative noise models (Ornstein-Uhlenbeck colored noise)

---

## License

MIT — see [LICENSE](LICENSE).

## References

- Lapicque, L. (1907) — *Recherches quantitatives sur l'excitation électrique des nerfs*
- Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. — *Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition*
- Dayan, P., & Abbott, L. F. — *Theoretical Neuroscience: Computational and Mathematical Modeling of Neural Systems*
