"""
Leaky Integrate-and-Fire (LIF) Neuron Simulation
=================================================
The LIF model is one of the simplest biologically plausible neuron models.
The key idea: a neuron accumulates input current like a leaky capacitor.
When membrane voltage crosses a threshold, it fires a spike and resets.

Governing equation:
    tau * dV/dt = -(V - V_rest) + R * I(t)

where:
    V       = membrane potential (mV)
    tau     = membrane time constant (ms) — how fast voltage decays
    V_rest  = resting potential (mV)
    R       = membrane resistance (MOhm)
    I(t)    = input current (nA)

Connection to the cerebellum:
    Granule cells in the cerebellar cortex are well-modelled by LIF dynamics.
    Their sparse firing (1-5% active at any time) emerges partly from the
    balance between leak and input — exactly what this simulation explores.
"""

import numpy as np
import matplotlib.pyplot as plt


# ── Neuron parameters (biologically grounded values) ──────────────────────
tau      = 20.0   # membrane time constant (ms)
V_rest   = -70.0  # resting potential (mV)
V_thresh = -50.0  # spike threshold (mV)
V_reset  = -75.0  # reset potential after spike (mV) — brief hyperpolarisation
R        = 10.0   # membrane resistance (MOhm)
t_ref    = 2.0    # refractory period (ms) — neuron cannot fire again immediately

# ── Simulation parameters ─────────────────────────────────────────────────
dt       = 0.1    # timestep (ms)
T        = 200.0  # total simulation time (ms)
time     = np.arange(0, T, dt)

# ── Input current: step current that turns on at 50ms and off at 150ms ────
I = np.zeros(len(time))
I[(time >= 50) & (time <= 150)] = 2.5   # nA — enough to drive firing


def simulate_lif(I, tau, V_rest, V_thresh, V_reset, R, t_ref, dt):
    """
    Simulate a single LIF neuron using Euler integration.
    Returns membrane voltage trace and spike times.
    """
    V          = np.zeros(len(I))
    V[0]       = V_rest
    spikes     = []
    ref_count  = 0          # counts down refractory timesteps

    for t in range(1, len(I)):
        if ref_count > 0:
            # During refractory period: hold at reset, don't integrate
            V[t] = V_reset
            ref_count -= 1
        else:
            # Euler step: dV = dt/tau * (-(V - V_rest) + R*I)
            dV   = (dt / tau) * (-(V[t-1] - V_rest) + R * I[t-1])
            V[t] = V[t-1] + dV

            # Threshold crossing → spike
            if V[t] >= V_thresh:
                V[t]      = 40.0          # draw spike peak for visualisation
                spikes.append(t * dt)
                ref_count = int(t_ref / dt)

    return V, spikes


V, spikes = simulate_lif(I, tau, V_rest, V_thresh, V_reset, R, t_ref, dt)

# ── Plot ──────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig.suptitle("Leaky Integrate-and-Fire Neuron", fontsize=13, fontweight='bold')

# Membrane voltage
ax1.plot(time, V, color='steelblue', linewidth=0.9, label='Membrane potential')
ax1.axhline(V_thresh, color='crimson', linestyle='--',
            linewidth=0.8, label=f'Threshold ({V_thresh} mV)')
ax1.axhline(V_rest,   color='gray',   linestyle=':',
            linewidth=0.8, label=f'Rest ({V_rest} mV)')
ax1.set_ylabel('Membrane potential (mV)')
ax1.legend(fontsize=8, loc='upper right')
ax1.set_ylim(-85, 50)

# Input current
ax2.plot(time, I, color='darkorange', linewidth=1.2, label='Input current')
ax2.set_ylabel('Input current (nA)')
ax2.set_xlabel('Time (ms)')
ax2.legend(fontsize=8)

# Mark spikes on voltage trace
for s in spikes:
    ax1.axvline(s, color='crimson', alpha=0.3, linewidth=0.6)

plt.tight_layout()
plt.savefig('lif_neuron.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSimulation complete.")
print(f"Number of spikes: {len(spikes)}")
print(f"Spike times (ms): {[round(s, 1) for s in spikes]}")
if len(spikes) > 1:
    isis = np.diff(spikes)
    print(f"Mean inter-spike interval: {isis.mean():.1f} ms")
    print(f"Mean firing rate: {1000/isis.mean():.1f} Hz")
