"""Finite sampling of the legacy crop pair conditioned on fitting the mixture.

For a repeated source of N samples, the legacy sampler draws start S uniformly
over 0..min((N-1)//2, N-minimum), then end E uniformly over
max(N//2+1, S+minimum)..N. Thus a feasible (S,E) has weight
1 / (N-max(N//2+1, S+minimum)+1), ignoring the constant start normalizer.

Sum these weights for each length using prefix sums. Draw length A with weight
wA[a] * sum(wB[:budget-a+1]), then B within its remaining budget, then each
start conditional on its length. This is the same joint conditional law as
successful legacy rejection sampling, without a random failure limit.
No additional truncation, source replacement, or duration cap per event.
"""
import numpy as np

from datasets.mix_utils import (_waveform, remove_zero_energy_samples,
                                repeat_to_min_duration, SAMPLE_RATE)

MINIMUM = SAMPLE_RATE // 2
POLICY = 'legacy_conditional_joint_fit_v1'


def length_weights(n, maximum, minimum=MINIMUM):
    """Unnormalized legacy length PMF, with bounded O(maximum) memory/time."""
    if n < minimum or minimum < 1 or maximum < minimum:
        raise ValueError('No feasible legacy crop length')
    last_start = min((n-1)//2, n-minimum)
    first_start = max(0, n//2+1-maximum)
    starts = np.arange(first_start, last_start+1, dtype=np.int64)
    weights = 1. / (n - np.maximum(n//2+1, starts+minimum) + 1)
    prefix = np.r_[0., np.cumsum(weights, dtype=np.float64)]
    lengths = np.arange(minimum, min(n, maximum)+1, dtype=np.int64)
    low = np.maximum(0, n//2+1-lengths)
    high = np.minimum(last_start, n-lengths)
    valid = low <= high
    result = np.zeros(maximum+1, dtype=np.float64)
    result[lengths[valid]] = (prefix[high[valid]-first_start+1] -
                              prefix[low[valid]-first_start])
    return result


def weighted_index(weights, rng):
    cdf = np.cumsum(weights, dtype=np.float64)
    total = cdf[-1]
    if not np.isfinite(total) or total <= 0:
        raise ValueError('No feasible conditional crop pair')
    draw = min(float(rng.random()) * total, np.nextafter(total, 0.))
    return int(np.searchsorted(cdf, draw, side='right'))


def sample_lengths(lengths, budget, rng, minimum=MINIMUM):
    if len(lengths) != 2 or budget < 2*minimum:
        raise ValueError('Two minimum-duration crops must fit the available budget')
    wa, wb = [length_weights(n, budget, minimum) for n in lengths]
    a = weighted_index(wa * np.cumsum(wb)[budget-np.arange(budget+1)], rng)
    b = weighted_index(wb[:budget-a+1], rng)
    return a, b


def _preprocess(waveform):
    original = _waveform(waveform)
    nonzero = remove_zero_energy_samples(original)
    repeated, copies = repeat_to_min_duration(nonzero)
    return repeated, dict(fsd_original_num_samples=len(original), num_nonzero_samples=len(nonzero),
        num_repeats=copies, additional_repeats=copies-1, repeated_num_samples=len(repeated))


def _geometry(base, start, end):
    return dict(**base, sampled_crop_start_sample=start, sampled_crop_end_sample=end,
        sampled_crop_num_samples=end-start, crop_start_sample=start, crop_end_sample=end,
        crop_num_samples=end-start, truncated_num_samples=0)


def conditional_pair_crops(waveforms, *, budget, rng):
    prepared = [_preprocess(w) for w in waveforms]
    lengths = sample_lengths([len(w) for w, _ in prepared], budget, rng)
    result = []
    for (waveform, base), length in zip(prepared, lengths):
        n = len(waveform)
        low = max(0, n//2+1-length)
        high = min((n-1)//2, n-MINIMUM, n-length)
        starts = np.arange(low, high+1, dtype=np.int64)
        weights = 1. / (n - np.maximum(n//2+1, starts+MINIMUM) + 1)
        start = int(starts[weighted_index(weights, rng)])
        end = start + length
        pad_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        result.append((waveform[start:end].copy(), _geometry(base, start, end), pad_seed))
    return result


def replay_conditional_crop(waveform, event):
    """Audit from explicit geometry, independently of conditional sampling RNG."""
    repeated, base = _preprocess(waveform)
    n = len(repeated)
    start, end = event['crop_start_sample'], event['crop_end_sample']
    if (not 0 <= start <= min((n-1)//2, n-MINIMUM) or
            not max(n//2+1, start+MINIMUM) <= end <= n):
        raise ValueError('Persisted crop is outside legacy crop support')
    return repeated[start:end].copy(), _geometry(base, start, end)
