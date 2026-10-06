"""Sparse conductance LIF model. Parameters are hypotheses, not measured BANC physiology.

Units: mV, ms, pF, nS, pA internally; public dt is seconds.
All outgoing synapses of one source share receptor kinetics and reversal class.
This permits a per-source saturated gate plus sparse matrix multiplication.
It does NOT implement receptor-specific co-transmission or neuromodulation.
"""
import numpy as np
from scipy.sparse import coo_matrix


class NeuralModel:
    def __init__(self, ids, pre, post, counts, source_class, params):
        self.ids = list(ids)
        self.index = {value: i for i, value in enumerate(ids)}
        self.p = params
        n = len(ids)
        pre, post = np.asarray(pre, int), np.asarray(post, int)
        counts = np.asarray(counts, dtype=float)
        if not np.all(np.isfinite(counts)) or np.any(counts <= 0):
            raise ValueError('Synapse counts must be finite and positive.')
        source_class = np.asarray(source_class)
        matrices = []
        for kind in ('exc', 'inh'):
            mask = source_class[pre] == kind
            weight = counts[mask] * params['unit_conductance_nS']
            matrix = coo_matrix((weight, (post[mask], pre[mask])), shape=(n, n)).tocsr()
            matrix.sum_duplicates()
            matrices.append(matrix)
        self.exc, self.inh = matrices
        self.v = np.full(n, params['rest_mV'], dtype=float)
        self.gate = np.zeros(n)
        self.refractory_ms = np.zeros(n)
        self.rates_hz = np.zeros(n)

    def indices(self, ids):
        missing = [x for x in ids if x not in self.index]
        if missing:
            raise ValueError(f'Configured neuron IDs not in loaded network: {missing[:10]}')
        return np.asarray([self.index[x] for x in ids], dtype=int)

    def step(self, current_pA, dt_s):
        p = self.p
        h = dt_s * 1000.0
        if not 0 < h <= 1.0:
            raise ValueError('Use neural timesteps in (0, 1] ms; 0.1 ms is the default.')
        # Existing release gates contribute now; new spikes act next timestep.
        self.gate *= np.exp(-h / p['synapse_decay_ms'])
        ge, gi = self.exc @ self.gate, self.inh @ self.gate
        leak = p['leak_nS']
        total = leak + ge + gi
        equilibrium = (leak * p['rest_mV'] + ge * p['exc_reversal_mV']
                       + gi * p['inh_reversal_mV'] + current_pA) / total
        ready = self.refractory_ms <= 0
        self.refractory_ms = np.maximum(0, self.refractory_ms - h)
        # Exact passive voltage update with conductances held fixed over this step.
        self.v[ready] = equilibrium[ready] + (self.v[ready] - equilibrium[ready]) * np.exp(
            -h * total[ready] / p['capacitance_pF'])
        self.v[~ready] = p['reset_mV']
        spikes = ready & (self.v >= p['threshold_mV'])
        self.v[spikes] = p['reset_mV']
        self.refractory_ms[spikes] = p['refractory_ms']
        # Saturating receptor recruitment, bounded in [0,1].
        self.gate[spikes] += (1 - self.gate[spikes]) * p['release_fraction']
        tau_s = p['rate_filter_ms'] / 1000
        self.rates_hz *= np.exp(-dt_s / tau_s)
        self.rates_hz[spikes] += 1 / tau_s
        if not np.isfinite(self.v).all():
            raise FloatingPointError('Non-finite neural voltage: check parameters.')
        return spikes


class MotorDecoder:
    def __init__(self, brain, populations, params):
        self.groups = [brain.indices(populations[k]) for k in ('output_left', 'output_right')]
        self.p = params

    def read(self, rates):
        population_rates = np.array([rates[g].mean() for g in self.groups])
        drive = np.asarray(self.p['baseline']) + np.asarray(self.p['gain_per_hz']) * population_rates
        return np.clip(drive, self.p['minimum'], self.p['maximum']), population_rates