# utils.py
# --------------------------------
# Utilities for building and running XOR neural network experiments.
#
# Network logic:
#   - E fires when exactly one of (A, B) fires (XOR condition)
#   - C acts as AND gate (fires when both A and B fire)
#   - D acts as OR gate (fires when either A or B fires)
#   - E = D AND NOT(C) = XOR(A, B)
#
# Typical workflow:
#   trials = build_in_domain_trials(trials_per_type=50, trial_length_ms=100)
#   for idx, tr in enumerate(trials):
#       net, a_bit, b_bit, a_time, b_time = run_single_in_domain_trial(tr, 100)
#       writer.write_trial(net, idx, 100, a_bit, b_bit, a_time, b_time)

import numpy as np
from NeuronNetwork import NeuronNetwork
from typing import Dict, List, Optional, Tuple, Union


def generate_random_stimulation_times_ms(
    num_stims: int, 
    max_time_ms: float, 
    min_time_between_stim_ms: float
) -> List[int]:
    """
    DEPRECATED: Use build_in_domain_trials() for new experiments.
    
    Generates random stimulus times across a continuous run.
    Kept for backward compatibility with legacy code.

    Args:
        num_stims: Number of stimuli to generate
        max_time_ms: Total duration
        min_time_between_stim_ms: Minimum separation between stimuli
        
    Returns:
        List of stimulus times in ascending order
    """
    stimulation_times = []
    current_time = 0
    time_per_stim_slot = max_time_ms / num_stims

    for i in range(num_stims):
        min_current_stim_time = current_time + min_time_between_stim_ms
        
        upper_limit_from_distribution = int((i + 1) * time_per_stim_slot)
        remaining_stims_to_place = num_stims - (i + 1)
        upper_limit_from_remaining_space = max_time_ms - (remaining_stims_to_place * min_time_between_stim_ms)
        
        upper_bound = min(upper_limit_from_distribution, upper_limit_from_remaining_space)
        lower_bound = max(min_current_stim_time, current_time)

        if lower_bound > upper_bound:
            return stimulation_times

        if lower_bound == upper_bound:
            stim_time = lower_bound
        else:
            stim_time = np.random.randint(lower_bound, upper_bound + 1)

        stimulation_times.append(int(stim_time))
        current_time = stim_time

    return stimulation_times


def build_in_domain_trials(
    trials_per_type: int,
    trial_length_ms: int = 100,
    start_guard_ms: Optional[int] = 5,
    end_guard_ms: Optional[int] = 39,
    stim_guard_ms: Optional[int] = None,  # Legacy symmetric mode
    rng_seed: Optional[int] = 42,
) -> List[Dict[str, Union[int, None]]]:
    """
    Build balanced trials for all 4 input patterns: (0,0), (0,1), (1,0), (1,1).
    
    Creates equal representation of all input combinations for XOR testing.
    Each trial places stimulus times randomly within guard boundaries.

    Guard modes:
      - Asymmetric (default): start_guard_ms=5, end_guard_ms=39
        → For 100ms trial: stim window [5, 60]ms, ~40ms response time
      - Symmetric (legacy): end_guard_ms=None, uses stim_guard_ms

    Args:
        trials_per_type: Trials per pattern (total = 4 × trials_per_type)
        trial_length_ms: Duration of each trial (typically 100)
        start_guard_ms: Earliest stimulus time
        end_guard_ms: Buffer from trial end (None for symmetric mode)
        stim_guard_ms: Used only in symmetric mode
        rng_seed: Random seed (None for non-deterministic)

    Returns:
        List of trial dicts with:
          - a_bit, b_bit: Input bits (0 or 1)
          - a_time, b_time: Stimulus times (ms) or None if bit=0
          
    Example:
        trials = build_in_domain_trials(50)  # 200 total trials
        # {'a_bit': 1, 'b_bit': 0, 'a_time': 37, 'b_time': None}
    """
    if trials_per_type <= 0:
        raise ValueError(f"trials_per_type must be positive")
    if trial_length_ms <= 0:
        raise ValueError(f"trial_length_ms must be positive")
    
    # Calculate stimulus window
    if end_guard_ms is None:
        # Symmetric mode
        g = stim_guard_ms if stim_guard_ms is not None else 5
        low, high = int(g), int(trial_length_ms - 1 - g)
    else:
        # Asymmetric mode
        sg = 5 if start_guard_ms is None else int(start_guard_ms)
        low, high = sg, int(trial_length_ms - 1 - end_guard_ms)

    if not (0 <= low <= high < trial_length_ms):
        raise ValueError(f"Invalid stimulus window [{low}, {high}] for trial_length={trial_length_ms}")

    rng = np.random.default_rng(rng_seed) if rng_seed is not None else np.random.default_rng()

    # Generate balanced trials
    trials = []
    for a_bit, b_bit in [(0,0), (0,1), (1,0), (1,1)]:
        for _ in range(trials_per_type):
            a_time = int(rng.integers(low, high + 1)) if a_bit == 1 else None
            b_time = int(rng.integers(low, high + 1)) if b_bit == 1 else None
            
            trials.append({
                'a_bit': a_bit, 
                'b_bit': b_bit, 
                'a_time': a_time, 
                'b_time': b_time
            })

    rng.shuffle(trials)
    return trials


def run_single_in_domain_trial(
    trial_def: Dict[str, Union[int, None]],
    trial_length_ms: int,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[NeuronNetwork, int, int, Optional[int], Optional[int]]:
    """
    Simulate one trial of the XOR neural circuit.
    
    Network topology:
    
        A ──┬──[0.7]──> C ──[-1.0]──┐
            │                       ↓
            └──[1.0]──> D ──[1.0]──> E (XOR output)
            ┌──[1.0]──↗
            │
        B ──┴──[0.7]──> C
    
    Logic:
      - C = AND: fires when BOTH inputs active (0.7+0.7 > threshold)
      - D = OR: fires when EITHER input active
      - E = XOR: D excites, C inhibits → fires for exactly one input

    Args:
        trial_def: Trial dict with {a_bit, b_bit, a_time, b_time}
        trial_length_ms: Simulation duration
        weights: Optional weight overrides (keys: A_to_C, B_to_C, etc.)
                
    Returns:
        (network, a_bit, b_bit, a_time, b_time)
        
    Example:
        trial = {'a_bit': 1, 'b_bit': 0, 'a_time': 25, 'b_time': None}
        net, a, b, at, bt = run_single_in_domain_trial(trial, 100)
        e_spikes = net.get_neuron_spike_times_ms('Neuron_E')
    """
    # Default weights tuned for XOR logic
    w = {
        'A_to_C': 0.7,   # Subthreshold alone
        'B_to_C': 0.7,   # Subthreshold alone
        'A_to_D': 1.0,   # Suprathreshold
        'B_to_D': 1.0,   # Suprathreshold
        'D_to_E': 1.0,   # Excitatory
        'C_to_E': -1.0,  # Inhibitory
    }
    
    if weights:
        w.update(weights)

    # Create network
    nn = NeuronNetwork('InDomain_XOR_System')
    
    # Add neurons
    nn.add_neuron('Neuron_A')  # Input A
    nn.add_neuron('Neuron_B')  # Input B
    nn.add_neuron('Neuron_C')  # AND gate
    nn.add_neuron('Neuron_D')  # OR gate
    nn.add_neuron('Neuron_E')  # XOR output

    # Wire connections
    nn.add_neuron_connection('Neuron_A', 'Neuron_C', w['A_to_C'])
    nn.add_neuron_connection('Neuron_B', 'Neuron_C', w['B_to_C'])
    nn.add_neuron_connection('Neuron_A', 'Neuron_D', w['A_to_D'])
    nn.add_neuron_connection('Neuron_B', 'Neuron_D', w['B_to_D'])
    nn.add_neuron_connection('Neuron_D', 'Neuron_E', w['D_to_E'])
    nn.add_neuron_connection('Neuron_C', 'Neuron_E', w['C_to_E'])

    # Set stimulation
    a_time = trial_def['a_time']
    b_time = trial_def['b_time']
    nn.set_direct_stimulation_time_ms('Neuron_A', [] if a_time is None else [a_time])
    nn.set_direct_stimulation_time_ms('Neuron_B', [] if b_time is None else [b_time])

    # Run simulation
    nn.run_simulation(int(trial_length_ms), record_membrane_potential=True)

    return nn, trial_def['a_bit'], trial_def['b_bit'], a_time, b_time