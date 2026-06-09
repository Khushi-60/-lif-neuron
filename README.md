# -lif-neuron
LIF neuron simulation
# Leaky Integrate-and-Fire Neuron Simulation

A single-neuron simulation implementing the Leaky Integrate-and-Fire (LIF) model from scratch using NumPy and Euler integration — no specialist neuroscience libraries required.

Written as part of background preparation for PhD research in NeuroAI and Computational Neuroscience, exploring the relationship between biological neural circuit principles and artificial network design.

---

## The Model

The LIF model captures the essential dynamics of a real neuron with one equation:

```
tau * dV/dt = -(V - V_rest) + R * I(t)
```

The neuron integrates incoming current like a leaky capacitor. When the membrane voltage crosses a threshold, it fires a spike and resets — mimicking the action potential mechanism without modelling the ionic channels explicitly. A refractory period after each spike prevents immediate re-firing, consistent with biological observation.

### Parameters

| Parameter | Value | Biological meaning |
|---|---|---|
| `tau` | 20 ms | Membrane time constant — how fast voltage decays without input |
| `V_rest` | −70 mV | Resting potential |
| `V_thresh` | −50 mV | Spike threshold |
| `V_reset` | −75 mV | Post-spike reset (brief hyperpolarisation) |
| `R` | 10 MΩ | Membrane resistance |
| `t_ref` | 2 ms | Refractory period |

These values are grounded in standard cortical neuron measurements (Dayan & Abbott, 2001).

---

## Output

Running the script produces a two-panel figure:

- **Top panel** — membrane voltage trace with threshold and resting potential marked; spike times shown as vertical lines
- **Bottom panel** — the input current step (on from 50 ms, off at 150 ms)

Terminal output reports spike count, spike times, mean inter-spike interval, and mean firing rate.

---

## Connection to Cerebellar Sparsity

Granule cells in the cerebellar cortex — the most numerous neurons in the brain — are well-described by LIF dynamics. Their characteristic sparse firing (only 1–5% active at any moment; Chadderton et al., 2004) emerges from the balance between the membrane leak and the sparse excitatory input they receive from mossy fibres. This simulation is a first step toward understanding that balance computationally.

The broader research question this connects to: does the 1–5% activity sparsity of cerebellar granule cells represent a principled minimum complexity threshold for real-time physical prediction, and can that threshold be recovered in artificial networks?

---

## Usage

```bash
# Install dependencies
pip install numpy matplotlib

# Run
python lif_neuron.py
```

No other dependencies. Compatible with Python 3.8+.

---

## References

- Dayan, P., & Abbott, L.F. (2001). *Theoretical Neuroscience*. MIT Press.
- Chadderton, P., Margrie, T.W., & Häusser, M. (2004). Integration of quanta in cerebellar granule cells during sensory processing. *Nature*, 428, 856–860.
- Zador, A.M. (2019). A critique of pure learning and what artificial neural networks can learn from animal brains. *Nature Communications*, 10, 3770. [doi.org/10.1038/s41467-019-11786-6](https://doi.org/10.1038/s41467-019-11786-6)
