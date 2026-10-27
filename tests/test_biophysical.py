import numpy as np
import pytest

from fruit_fly_stuff.biophysical import Experiment, rates, simulate, save_results


def test_hh_rate_functions_are_finite_at_singular_voltages():
    for voltage in [-40.0, -55.0, -65.0]:
        assert all(np.all(np.isfinite(x)) for x in rates(np.array([voltage])))


def test_action_potentials_come_from_channel_dynamics():
    out = simulate()
    assert out["voltage_mv"].shape[1] == 3
    assert out["spikes"][:, 0].sum() >= 2
    assert np.max(out["voltage_mv"][:, 0]) > 20
    assert np.all(np.isfinite(out["voltage_mv"]))


def test_without_stimulation_neurons_rest():
    cfg = Experiment(injected_current=0.0)
    out = simulate(cfg, connected=False)
    assert not np.any(out["spikes"])
    assert np.max(np.abs(out["voltage_mv"] + 65.0)) < 2.0


def test_synaptic_coupling_changes_downstream_response():
    on = simulate(connected=True)
    off = simulate(connected=False)
    assert np.all(off["spikes"][:, 1:] == 0)
    assert not np.allclose(on["voltage_mv"][:, 1:], off["voltage_mv"][:, 1:])


def test_export_csv(tmp_path):
    result = simulate(Experiment(duration_ms=80.0))
    stats = save_results(result, tmp_path)
    assert stats["neuron_spike_counts"][0] > 0
    assert (tmp_path / "hh_neurons.csv").exists()
    assert (tmp_path / "summary.json").exists()


def test_large_timestep_rejected():
    with pytest.raises(ValueError, match="invalid"):
        Experiment(dt_ms=1.0)
