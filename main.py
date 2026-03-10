# importing the libraries 

import numpy as np
import matplotlib.pyplot as plt


# 1. DEFINING PARAMETERS

tau_m    = 20.0    # membrane time constant (ms)
V_rest   = -70.0   # resting potential      (mV)
V_thresh = -55.0   # spike threshold        (mV)
V_reset  = -75.0   # reset potential        (mV)
V_spike  =  40.0   # spike height for plot  (mV)
R_m      =  10.0   # membrane resistance    (MΩ)
t_ref    =   2.0   # refractory period      (ms)
sigma = 1.0        # noise strength (mV) 
dt = 0.1           # time step (ms)
T  = 500.0         # simulation duration (ms)

t  = np.arange(0, T, dt)
N  = len(t)

# Derived quantities
I_rheo = (V_thresh - V_rest) / R_m   # minimum current to fire (nA)
f_max  = 1000.0 / t_ref              # maximum firing rate (Hz)

print(f"Rheobase current : {I_rheo:.2f} nA")
print(f"Max firing rate  : {f_max:.0f} Hz")
print(f"Noise strength   : sigma = {sigma} mV")

# 2. DEFINING SIMULATION FUNCTION


def simulate_lif(I_ext, t, dt, tau_m, V_rest, V_thresh, V_reset, R_m, t_ref,
                 sigma=0.0): 

    N = len(t)
    V = np.zeros(N)
    V[0] = V_rest
    spike_times = []
    last_spike  = -np.inf

    for i in range(1, N):

        # Step 1: refractory check
        if t[i] - last_spike < t_ref:
            V[i] = V_reset
            continue

        # Step 2: Euler integration
        noise = np.random.normal(0, sigma) * np.sqrt(dt)       
        dV    = (dt / tau_m) * (-(V[i-1] - V_rest) + R_m * I_ext[i-1])
        V[i]  = V[i-1] + dV + noise                             

        # Step 3: threshold check
        if V[i] >= V_thresh:
            spike_times.append(t[i])
            last_spike = t[i]
            V[i] = V_reset

    return V, spike_times


def add_spike_peaks(V, spike_times, t, dt, V_spike=40.0):
    """Add visual spike markers to the voltage trace for plotting."""
    V_plot = V.copy()
    for ts in spike_times:
        idx = int(round(ts / dt))
        if idx < len(V_plot):
            V_plot[idx] = V_spike
    return V_plot



# 3. FIGURE 1 (VOLTAGE TRACES)
#    (Subthreshold, suprathreshold, and strong drive)

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
fig.suptitle(f"LIF Neuron — Voltage Traces  (noise σ = {sigma} mV)",
             fontsize=13, fontweight='bold')

cases = [
    (0.8 * I_rheo, '#5B8DB8', 'Subthreshold   (0.8 × I_rheo)'),
    (1.5 * I_rheo, '#E07B39', 'Suprathreshold (1.5 × I_rheo)'),
    (3.0 * I_rheo, '#4CAF7D', 'Strong drive   (3.0 × I_rheo)'),
]

for ax, (I_amp, color, label) in zip(axes, cases):

    # Step current: off [0-50ms], on [50-400ms], off [400-500ms]
    I_ext = np.zeros(N)
    I_ext[(t >= 50) & (t < 400)] = I_amp

    V, spikes = simulate_lif(I_ext, t, dt, tau_m, V_rest,
                              V_thresh, V_reset, R_m, t_ref,
                              sigma=sigma)  # ← pass sigma
    V_plot = add_spike_peaks(V, spikes, t, dt, V_spike)

    ax.plot(t, V_plot, color=color, lw=1.5)
    ax.axhline(V_thresh, color='red',  ls='--', lw=1.0, alpha=0.7, label='Threshold')
    ax.axhline(V_rest,   color='gray', ls=':',  lw=1.0, alpha=0.6, label='V_rest')
    ax.axvspan(50, 400, alpha=0.06, color=color)

    rate  = len(spikes) / 0.35   # Hz over 350 ms stimulus window
    info  = f"  |  {len(spikes)} spikes, {rate:.1f} Hz" if spikes else "  |  No spikes"
    ax.set_title(label + info, fontsize=10)
    ax.set_ylabel("V (mV)")
    ax.set_ylim(V_reset - 8, V_spike + 12)
    ax.legend(loc='upper right', fontsize=8, frameon=False)

axes[-1].set_xlabel("Time (ms)")
plt.tight_layout()
plt.savefig("fig1_voltage_traces.png", dpi=150, bbox_inches='tight')
plt.show()
print("Figure 1 saved.")


# 4. FIGURE 2 — SUBTHRESHOLD DYNAMICS
#    Panel A: brief pulse -> exponential decay -> measure tau_m
#    Panel B: step current -> exponential approach to V_ss


fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Subthreshold Dynamics & Membrane Time Constant",
             fontsize=13, fontweight='bold')

t2 = np.arange(0, 200.0, dt)
N2 = len(t2)

# PANEL A
I_pulse = np.zeros(N2)
I_pulse[(t2 >= 10) & (t2 < 12)] = 0.5   

V_pulse, _ = simulate_lif(I_pulse, t2, dt, tau_m, V_rest,
                           V_thresh, V_reset, R_m, t_ref,
                           sigma=0.0)  

# PANEL B
pk_idx   = np.argmax(V_pulse)
t_peak   = t2[pk_idx]
V_peak   = V_pulse[pk_idx]
t_dec    = t2[pk_idx:]
V_theory = V_rest + (V_peak - V_rest) * np.exp(-(t_dec - t_peak) / tau_m)

ax = axes[0]
ax.plot(t2, V_pulse, color='#5B8DB8', lw=2.0, label='Simulation')
ax.plot(t_dec, V_theory, 'r--', lw=1.5, alpha=0.85,
        label=f'Theory: exp(-t/τ_m),  τ_m = {tau_m} ms')

# Marking 1/e point
t_tau_pt = t_peak + tau_m
V_tau_pt = V_rest + (V_peak - V_rest) * np.exp(-1)
ax.scatter([t_tau_pt], [V_tau_pt], color='darkred', zorder=5, s=60)
ax.annotate(f'1/e point  (τ_m = {tau_m} ms)',
            xy=(t_tau_pt, V_tau_pt),
            xytext=(t_tau_pt + 12, V_tau_pt - 2),
            fontsize=9, color='darkred',
            arrowprops=dict(arrowstyle='->', color='darkred'))

ax.axhline(V_rest, color='gray', ls=':', lw=1.0, alpha=0.7)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('V (mV)')
ax.set_title('A — Exponential decay after brief pulse\n(how τ_m is measured experimentally)',
             fontsize=10)
ax.legend(fontsize=9, frameon=False)

# PANEL B
I_step = np.zeros(N2)
I_step[t2 >= 20] = 0.8 * I_rheo

V_step, _ = simulate_lif(I_step, t2, dt, tau_m, V_rest,
                          V_thresh, V_reset, R_m, t_ref,
                          sigma=0.0)   

V_ss        = V_rest + R_m * 0.8 * I_rheo
t_rise      = t2[t2 >= 20]
V_rise_theo = V_ss + (V_rest - V_ss) * np.exp(-(t_rise - 20) / tau_m)

ax = axes[1]
ax.plot(t2, V_step, color='#4CAF7D', lw=2.0, label='Simulation')
ax.plot(t_rise, V_rise_theo, 'r--', lw=1.5, alpha=0.85, label='Theory')
ax.axhline(V_ss,     color='purple', ls='-.', lw=1.2,
           label=f'V_ss = {V_ss:.1f} mV  (= V_rest + R_m·I)')
ax.axhline(V_thresh, color='red',    ls='--', lw=1.0, alpha=0.7,
           label='Threshold')
ax.axhline(V_rest,   color='gray',   ls=':',  lw=1.0, alpha=0.6)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('V (mV)')
ax.set_title('B — Exponential approach to steady state\n(subthreshold step current)',
             fontsize=10)
ax.legend(fontsize=8.5, frameon=False)

plt.tight_layout()
plt.savefig("fig2_subthreshold.png", dpi=150, bbox_inches='tight')
plt.show()
print("Figure 2 saved.")



# 5. FIGURE 3 — F-I CURVE

T_fi = 1000.0
t_fi = np.arange(0, T_fi, dt)

I_sweep      = np.linspace(0, 5 * I_rheo, 60)
firing_rates = []

for I_amp in I_sweep:
    I_ext = np.ones(len(t_fi)) * I_amp
_, spikes = simulate_lif(I_ext, t_fi, dt, tau_m, V_rest,
                          V_thresh, V_reset, R_m, t_ref,
                          sigma=sigma)

    firing_rates.append(len(spikes))

firing_rates = np.array(firing_rates)


def fi_analytical(I_arr, tau_m, R_m, V_rest, V_thresh, V_reset, t_ref):
    """Exact analytical F-I curve for constant current LIF (no noise)."""
    rates = []
    for I in I_arr:
        V_ss = V_rest + R_m * I
        if V_ss <= V_thresh:
            rates.append(0.0)
        else:
            T_isi = tau_m * np.log((V_ss - V_reset) / (V_ss - V_thresh)) + t_ref
            rates.append(1000.0 / T_isi)
    return np.array(rates)


I_th     = np.linspace(0, 5 * I_rheo, 400)
rates_th = fi_analytical(I_th, tau_m, R_m, V_rest, V_thresh, V_reset, t_ref)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(I_sweep, firing_rates, 'o', color='#5B8DB8', ms=6,
        alpha=0.9, label=f'Simulation (σ = {sigma} mV)')
ax.plot(I_th, rates_th, 'r-', lw=2.0, alpha=0.85,
        label='Analytical solution (no noise)')
ax.axvline(I_rheo, color='gray',   ls='--', lw=1.2,
           label=f'I_rheo = {I_rheo:.2f} nA')
ax.axhline(f_max,  color='purple', ls='-.', lw=1.2,
           label=f'f_max = {f_max:.0f} Hz  (= 1/t_ref)')
ax.set_xlabel('Injected current  I  (nA)', fontsize=11)
ax.set_ylabel('Firing rate  f  (Hz)', fontsize=11)
ax.set_title('F-I Curve — Simulation vs. Analytical', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, frameon=False)
ax.set_xlim(0, None)
ax.set_ylim(0, None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("fig3_fi_curve.png", dpi=150, bbox_inches='tight')
plt.show()
print("Figure 3 saved.")



# 6. SUMMARY STATISTICS

t_1s   = np.arange(0, 1000.0, dt)
I_1s   = np.ones(len(t_1s)) * 2.0 * I_rheo
V_1s, spk_1s = simulate_lif(I_1s, t_1s, dt, tau_m, V_rest,
                              V_thresh, V_reset, R_m, t_ref,
                              sigma=sigma)  # ← noise ON

isis     = np.diff(spk_1s) if len(spk_1s) > 1 else [0]
mean_isi = np.mean(isis)
cv_isi   = np.std(isis) / mean_isi if mean_isi > 0 else 0

print("\n" + "=" * 50)
print(f"SUMMARY  (I = 2 × I_rheo,  T = 1000 ms,  σ = {sigma} mV)")
print("=" * 50)
print(f"  Spike count     : {len(spk_1s)}")
print(f"  Firing rate     : {len(spk_1s)} Hz")
print(f"  Mean ISI        : {mean_isi:.2f} ms")
print(f"  ISI CV          : {cv_isi:.4f}  (no noise = 0, biological ≈ 1)")
print(f"  Max rate (1/t_ref): {f_max:.0f} Hz")
print(f"  Noise strength  : sigma = {sigma} mV")
print("=" * 50)
print("\nSaved: fig1_voltage_traces.png")
print("       fig2_subthreshold.png")
print("       fig3_fi_curve.png")
