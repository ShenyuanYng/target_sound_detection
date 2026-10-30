"""Two FSD stems with controlled interval overlap and unchanged RMS gains."""
import numpy as np

from datasets.active_rms import active_rms, mix_rms
from datasets.mix_utils import TARGET_NUM_SAMPLES, prepare_fsd_crop, compute_rms


class PlacementError(ValueError):
    pass


def place_pair(lengths, overlap, rng, fraction_range=(.25, .5)):
    """Uniform integer overlap length, random order and feasible translation.

    Reject impossible crops; never shorten or truncate a stem to make it fit.
    Overlap fraction is interval intersection / shorter crop length.
    """
    a, b = map(int, lengths)
    n = TARGET_NUM_SAMPLES
    if min(a, b) < 1 or max(a, b) > n:
        raise PlacementError('Invalid source lengths')
    order = [int(i) for i in rng.permutation(2)]
    starts = [0, 0]
    if overlap:
        low = max(1, int(np.ceil(min(a, b) * fraction_range[0])))
        high = int(np.floor(min(a, b) * fraction_range[1]))
        if low > high:
            raise PlacementError('No integer overlap length in the requested range')
        intersection = int(rng.integers(low, high + 1))
        span = a + b - intersection
        if span > n:
            raise PlacementError('Partially overlapping sources do not fit in 10 seconds')
        leading = int(rng.integers(0, n - span + 1))
        starts[order[0]] = leading
        starts[order[1]] = leading + lengths[order[0]] - intersection
    else:
        free = n - a - b
        if free < 0:
            raise PlacementError('Nonoverlapping sources do not fit in 10 seconds')
        cuts = np.sort(rng.choice(free + 2, size=2, replace=False))
        leading, middle = int(cuts[0]), int(cuts[1] - cuts[0] - 1)
        starts[order[0]] = leading
        starts[order[1]] = leading + lengths[order[0]] + middle
        intersection = 0
    return starts, intersection


def make_tse_dual_mixture(background, waveforms, *, seed, overlap, snr_db,
                         threshold_dbfs=-40., frame_ms=20., max_attempts=128,
                         fraction_range=(.25, .5), silence_rms=1e-5):
    if len(waveforms) != 2 or max_attempts < 1:
        raise ValueError('Need two sources and a positive resampling limit')
    rng = np.random.default_rng(seed)
    for attempt in range(max_attempts):
        crop_seeds = [int(v) for v in rng.integers(0, 2**63-1, size=2)]
        prepared = [prepare_fsd_crop(w, rng=s) for w, s in zip(waveforms, crop_seeds)]
        crops = [p[0] for p in prepared]
        try:
            starts, intersection = place_pair([len(c) for c in crops], overlap, rng, fraction_range)
        except PlacementError:
            continue
        # Whole-file weak labels do not localize a class; at least reject silent crops.
        if any(compute_rms(c) <= silence_rms for c in crops):
            continue
        active_masks = []
        for start, crop in zip(starts, crops):
            mask = active_rms(crop, threshold_dbfs=threshold_dbfs, frame_ms=frame_ms)[1]
            full = np.zeros(TARGET_NUM_SAMPLES, dtype=bool)
            full[start:start+len(crop)] = mask
            active_masks.append(full)
        active_overlap = int(np.count_nonzero(active_masks[0] & active_masks[1]))
        # A geometric overlap containing only silence is not an overlap training case.
        if overlap and active_overlap == 0:
            continue
        targets, events = [], []
        for (crop, geometry, pad_seed), crop_seed, start in zip(prepared, crop_seeds, starts):
            bg, temporary, _, scaling = mix_rms(background, crop, snr_db=snr_db,
                threshold_dbfs=threshold_dbfs, frame_ms=frame_ms,
                normalization='none', rng=pad_seed)
            scaled = temporary[scaling['target_start_sample']:scaling['target_end_sample']]
            target = np.zeros(TARGET_NUM_SAMPLES, dtype=np.float32)
            target[start:start+len(crop)] = scaled
            targets.append(target)
            events.append(dict(crop_seed=crop_seed, **geometry,
                target_start_sample=start, target_end_sample=start+len(crop),
                pad_left=start, pad_right=TARGET_NUM_SAMPLES-start-len(crop),
                target_gain=scaling['target_gain'], snr_db=float(snr_db),
                active_snr_db=scaling['snr_db_exported'], rms_fallback=scaling['rms_fallback']))
        mixture = bg + targets[0] + targets[1]
        divisor = max(1., float(np.abs(mixture).max()))
        bg, targets, mixture = bg/divisor, [t/divisor for t in targets], mixture/divisor
        if any(compute_rms(t) <= silence_rms for t in targets):
            continue
        error = float(np.max(np.abs(mixture - (bg + targets[0] + targets[1]))))
        if error > 1e-6:
            raise ValueError('Mixture/stem sum mismatch')
        return bg, targets, mixture, dict(seed=int(seed), snr_db=float(snr_db),
            overlap=bool(overlap), overlap_samples=intersection,
            overlap_fraction=intersection/min(map(len, crops)),
            simultaneous_active_samples=active_overlap, events=events,
            crop_attempts=attempt+1, final_normalization_divisor=divisor,
            stem_sum_max_error=error)
    raise PlacementError(f'Could not place two audible FSD crops after {max_attempts} attempts')
