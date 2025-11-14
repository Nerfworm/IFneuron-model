# utils.py
# --------------------------------
# Utilities for:
#   1) Legacy continuous random stim generation (kept for compatibility)
#   2) Building clean, balanced in-domain trials (00/01/10/11)
#   3) Running ONE trial of the 5-neuron XOR-ish network
#
# Typical usage pattern for in-domain dataset:
#   trials = build_in_domain_trials(trials_per_type=50, trial_length_ms=100)
#   for idx, tr in enumerate(trials):
#       net, a_bit, b_bit, a_time, b_time = run_single_in_domain_trial(tr, 100, weights=None)
#       writer.write_trial(net, idx, 100, a_bit, b_bit, a_time, b_time)

import numpy as np
from NeuronNetwork import NeuronNetwork

# ---------- (1) Legacy generator: continuous random schedule ----------
def generate_random_stimulation_times_ms(num_stims: int, max_time_ms: float, min_time_between_stim_ms: float) -> list:
    """
    LEGACY FUNCTION (kept so older scripts don't break).
    Generates an ascending list of random stimulus times across a *continuous*
    run of 'max_time_ms', trying to keep at least 'min_time_between_stim_ms'
    between successive stims.

    NOTE: Not used for in-domain trial datasets (those are trial-based).
    """
    stimulation_times = []
    current_time = 0
    time_per_stim_slot = max_time_ms / num_stims

    for i in range(num_stims):
        min_current_stim_time = current_time + min_time_between_stim_ms

        # Upper bound from proportional partitioning of the run
        upper_limit_from_distribution = int((i + 1) * time_per_stim_slot)

        # Ensure we leave room (at least min interval) for the remaining stims
        remaining_stims_to_place = num_stims - (i + 1)
        upper_limit_from_remaining_space = max_time_ms - (remaining_stims_to_place * min_time_between_stim_ms)

        # Final bounds for this stim
        upper_bound = min(upper_limit_from_distribution, upper_limit_from_remaining_space)
        lower_bound = max(min_current_stim_time, current_time)

        # If bounds invalid, return what we've built so far
        if lower_bound > upper_bound:
            return stimulation_times

        # Choose exactly or randomly within bounds
        if lower_bound == upper_bound:
            stim_time = lower_bound
        else:
            stim_time = np.random.randint(lower_bound, upper_bound + 1)

        stimulation_times.append(int(stim_time))
        current_time = stim_time

    return stimulation_times


# ---------- (2) In-domain, balanced trial schedule ----------
def build_in_domain_trials(
    trials_per_type: int,
    trial_length_ms: int = 100,
    # New defaults: latest stim ≈ 60 ms in a 100 ms trial
    start_guard_ms: int | None = 5,
    end_guard_ms: int | None = 39,
    # Back-compat: if you set end_guard_ms=None, we’ll use this symmetric guard
    stim_guard_ms: int | None = None,
    rng_seed: int | None = 42,
):
    """
    Build a balanced list of trials for the 4 in-domain input patterns:
      (A,B) in {(0,0), (0,1), (1,0), (1,1)}.

    Each trial:
      - Length = 'trial_length_ms' ms
      - If a bit is 1, place ONE stimulus time within [low, high] (inclusive).
      - If a bit is 0, there is no stimulus for that channel.

    Guard policy:
      - By default we use asymmetric guards: start_guard_ms=5, end_guard_ms=39
        → with trial_length=100, the stim window becomes 5..60 ms.
      - If you pass end_guard_ms=None, we fall back to a symmetric guard:
        low = stim_guard_ms, high = trial_length-1-stim_guard_ms.
      - You can override any of these from the notebook per run.

    Returns:
      trials: list of dicts with keys {a_bit,b_bit,a_time,b_time}, shuffled.
    """
    # Choose guard mode
    if end_guard_ms is None:
        # Symmetric mode (backward compatible)
        g = stim_guard_ms if stim_guard_ms is not None else 5
        low  = int(g)
        high = int(trial_length_ms - 1 - g)
    else:
        # Asymmetric mode (preferred)
        sg = 5 if start_guard_ms is None else int(start_guard_ms)
        eg = int(end_guard_ms)
        low  = sg
        high = int(trial_length_ms - 1 - eg)

    if not (0 <= low <= high < trial_length_ms):
        raise ValueError(
            f"Invalid guards for trial_length={trial_length_ms}: "
            f"low={low}, high={high} (start_guard={start_guard_ms}, end_guard={end_guard_ms}, stim_guard={stim_guard_ms})"
        )

    # RNG
    rng = np.random.default_rng(rng_seed) if rng_seed is not None else np.random.default_rng()

    trials = []
    combos = [(0,0), (0,1), (1,0), (1,1)]
    for a_bit, b_bit in combos:
        for _ in range(trials_per_type):
            a_time = int(rng.integers(low, high + 1)) if a_bit == 1 else None
            b_time = int(rng.integers(low, high + 1)) if b_bit == 1 else None
            trials.append({'a_bit': a_bit, 'b_bit': b_bit, 'a_time': a_time, 'b_time': b_time})

    rng.shuffle(trials)
    return trials


# ---------- (3) Run one in-domain trial on the XOR-ish 5-neuron circuit ----------
def run_single_in_domain_trial(
    trial_def: dict,
    trial_length_ms: int,
    weights: dict | None = None,
):
    """
    Build the network and simulate ONE trial.

    Network topology (matches your original):
      A -> C (exc)
      B -> C (exc)
      A -> D (exc)
      B -> D (exc)
      D -> E (exc)
      C -> E (inh)

    Args:
      trial_def       : dict from build_in_domain_trials (a_bit/b_bit/a_time/b_time)
      trial_length_ms : length of this trial window
      weights         : optional dict to override default weights

    Returns:
      (nn, a_bit, b_bit, a_time, b_time)
        nn      : simulated NeuronNetwork for this trial
        a_bit   : 0/1
        b_bit   : 0/1
        a_time  : int ms or None
        b_time  : int ms or None
    """
    # Default weights (you can tweak via 'weights' arg if desired)
    w = {
        'A_to_C': 0.7,
        'B_to_C': 0.7,
        'A_to_D': 1.0,
        'B_to_D': 1.0,
        'D_to_E': 1.0,
        'C_to_E': -1.0,
    }
    if weights:
        w.update(weights)

    # Create network and neurons
    nn = NeuronNetwork('InDomain_XOR_System')
    nn.add_neuron('Neuron_A')
    nn.add_neuron('Neuron_B')
    nn.add_neuron('Neuron_C')
    nn.add_neuron('Neuron_D')
    nn.add_neuron('Neuron_E')

    # Wire connections
    nn.add_neuron_connection('Neuron_A', 'Neuron_C', w['A_to_C'])
    nn.add_neuron_connection('Neuron_B', 'Neuron_C', w['B_to_C'])
    nn.add_neuron_connection('Neuron_A', 'Neuron_D', w['A_to_D'])
    nn.add_neuron_connection('Neuron_B', 'Neuron_D', w['B_to_D'])
    nn.add_neuron_connection('Neuron_D', 'Neuron_E', w['D_to_E'])
    nn.add_neuron_connection('Neuron_C', 'Neuron_E', w['C_to_E'])

    # Assign direct stim times (0 or 1 per channel within this trial)
    a_time = trial_def['a_time']
    b_time = trial_def['b_time']
    nn.set_direct_stimulation_time_ms('Neuron_A', [] if a_time is None else [a_time])
    nn.set_direct_stimulation_time_ms('Neuron_B', [] if b_time is None else [b_time])

    # Run exactly one trial window
    nn.run_simulation(int(trial_length_ms), record_membrane_potential=True)

    # Return network and the ground-truth input bits/times for this trial
    return nn, trial_def['a_bit'], trial_def['b_bit'], a_time, b_time
