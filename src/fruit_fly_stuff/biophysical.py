"""Classical Hodgkin–Huxley sodium/potassium membrane model for 3 cells.

These *biophysical* equations originate from squid giant axon work.
They illustrate how real neurons generate action potentials, but are NOT
calibrated to any particular fruit-fly neuron type.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

# Classical HH constants: mV, ms, uA/cm^2, mS/cm^2.
C_M = 1.0
G_NA, G_K, G_L = 120.0, 36.0, 0.3
E_NA, E_K, E_L = 50.0, -77.0, -54.387
E_EXC, E_INH = 0.0, -80.0
SYN_TAU_MS = 5.0


@dataclass(frozen=True)
class Experiment:
    duration_ms: float = 120.0
    dt_ms: float = 0.025
    pulse_from_ms: float = 10.0
    pulse_until_ms: float = 70.0
    injected_current: float = 15.0

    def __post_init__(self):
        values = (self.duration_ms, self.dt_ms, self.pulse_from_ms,
                  self.pulse_until_ms, self.injected_current)
        if not all(np.isfinite(x) for x in values):
            raise ValueError("parameters must be finite")
        if not (0 < self.dt_ms <= 0.05 and self.duration_ms > self.dt_ms
                and 0 <= self.pulse_from_ms < self.pulse_until_ms <= self.duration_ms
                and self.injected_current >= 0):
            raise ValueError("invalid timestep, duration, injection window or current")


def _vtrap(x: np.ndarray, y: float) -> np.ndarray:
    """Stable x / (exp(x/y)-1), using the limiting value at x=0."""
    return np.where(
        np.abs(x / y) < 1e-6,
        y * (1.0 - x / (2.0 * y)),
        x / np.expm1(x / y),
    )


def rates(v: np.ndarray) -> tuple[np.ndarray, ...]:
    """Voltage-dependent HH alpha/beta gates (units 1/ms)."""
    am = 0.1 * _vtrap(-(v + 40.0), 10.0)
    bm = 4.0 * np.exp(-(v + 65.0) / 18.0)
    ah = 0.07 * np.exp(-(v + 65.0) / 20.0)
    bh = 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))
    an = 0.01 * _vtrap(-(v + 55.0), 10.0)
    bn = 0.125 * np.exp(-(v + 65.0) / 80.0)
    return am, bm, ah, bh, an, bn


def simulate(config: Experiment | None = None, *,
             connected: bool = True) -> dict[str, np.ndarray]:
    """Simulate 3 membrane models with directed conductance synapses.

    Cell 0 receives a current step and excites cells 1 and 2.
    Cell 2 inhibits cell 1. In the control condition, all synapses are off.
    """
    cfg = config or Experiment()
    steps = int(round(cfg.duration_ms / cfg.dt_ms))
    time_ms = np.arange(steps + 1, dtype=np.float64) * cfg.dt_ms
    voltage = np.zeros((steps + 1, 3), dtype=np.float64)
    voltage[0] = -65.0
    spikes = np.zeros((steps + 1, 3), dtype=bool)
    exc = np.zeros(3, dtype=np.float64)
    inh = np.zeros(3, dtype=np.float64)

    am, bm, ah, bh, an, bn = rates(voltage[0])
    m = am / (am + bm)
    h = ah / (ah + bh)
    n = an / (an + bn)
    decay = np.exp(-cfg.dt_ms / SYN_TAU_MS)

    for k in range(1, steps + 1):
        t = time_ms[k - 1]
        v = voltage[k - 1]
        am, bm, ah, bh, an, bn = rates(v)

        # Euler integration at 0.025 ms; gate kinetics use prior voltage.
        ionic = G_NA * m**3 * h * (v - E_NA) + G_K * n**4 * (v - E_K) + G_L * (v - E_L)
        exc *= decay
        inh *= decay
        synaptic = exc * (v - E_EXC) + inh * (v - E_INH)
        current = np.zeros(3)
        if cfg.pulse_from_ms <= t < cfg.pulse_until_ms:
            current[0] = cfg.injected_current

        new_v = v + cfg.dt_ms * ((current - ionic - synaptic) / C_M)
        m += cfg.dt_ms * (am * (1.0 - m) - bm * m)
        h += cfg.dt_ms * (ah * (1.0 - h) - bh * h)
        n += cfg.dt_ms * (an * (1.0 - n) - bn * n)

        if not (np.all(np.isfinite(new_v)) and np.all(np.isfinite(m))
                and np.all(np.isfinite(h)) and np.all(np.isfinite(n))):
            raise FloatingPointError("unstable integration; choose smaller dt")
        spike = (v < 0) & (new_v >= 0)
        if connected:
            if spike[0]:
                exc[1] += 1.8
                exc[2] += 1.8
            if spike[2]:
                inh[1] += 2.5
        spikes[k] = spike
        voltage[k] = new_v
    return {"time_ms": time_ms, "voltage_mv": voltage, "spikes": spikes}


def save_results(result: dict[str, np.ndarray], output_dir: Path, *,
                 plot: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    times, volts, spikes = result["time_ms"], result["voltage_mv"], result["spikes"]
    csv_file = output_dir / "hh_neurons.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["time_ms", "V_stimulated_mv", "V_excitatory_mv",
                         "V_inhibitory_mv", "spike0", "spike1", "spike2"])
        for t, v, s in zip(times, volts, spikes):
            writer.writerow([f"{t:.5f}", *(f"{x:.5f}" for x in v),
                             *(int(x) for x in s)])
    stats = {
        "model": "3-cell classical Hodgkin-Huxley with conductance synapses (not fly-specific)",
        "time_ms": float(times[-1]),
        "neuron_spike_counts": [int(v) for v in spikes.sum(axis=0)],
        "csv": str(csv_file),
    }
    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise RuntimeError("Plotting requires: python -m pip install -e '.[plots]'") from error
        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        labels = ["Neuron 0: injected current", "Neuron 1: downstream target",
                  "Neuron 2: inhibitory neuron"]
        for i, axis in enumerate(axes):
            axis.plot(times, volts[:, i], linewidth=1.0)
            axis.set(ylabel="mV", title=labels[i])
            axis.grid(alpha=0.2)
        axes[-1].set_xlabel("Time (ms)")
        fig.tight_layout()
        png_file = output_dir / "hh_neurons.png"
        fig.savefig(png_file, dpi=150)
        plt.close(fig)
        stats["plot"] = str(png_file)
    (output_dir / "summary.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("results/biophysical"))
    p.add_argument("--duration-ms", type=float, default=120.0)
    p.add_argument("--current", type=float, default=15.0)
    p.add_argument("--no-synapses", action="store_true",
                   help="Control run: uncouple all three neurons")
    p.add_argument("--plot", action="store_true", help="Write voltage-trace PNG")
    args = p.parse_args()
    cfg = Experiment(duration_ms=args.duration_ms, pulse_until_ms=min(args.duration_ms, 70.0),
                     injected_current=args.current)
    result = simulate(cfg, connected=not args.no_synapses)
    print(json.dumps(save_results(result, args.output, plot=args.plot), indent=2))


if __name__ == "__main__":
    main()
