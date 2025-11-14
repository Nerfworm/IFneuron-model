# NeuronNetwork.py
# ---------------------------------
# A tiny container that:
#  - Holds multiple IFneurons
#  - Wires synaptic connections between them
#  - Runs a simulation loop (1 ms steps)
#  - Provides helpers to read spike trains / Vm traces
#  - Provides helpers to read stimulus times/vectors (for GT writer)

import IFneuron
import numpy as np

class NeuronNetwork:
    """
    Network of IFneurons with convenience accessors for simulation outputs.
    """
    def __init__(self, id: 'str'):
        self.id = id
        self.all_neurons = {}     # dict: neuron_id -> IFneuron
        self._run_time_len = 0    # number of ms simulated in the last run

    # -------- Building the network --------
    def add_neuron(self, neuron_id: str):
        """
        Create a new neuron with given id (must be unique within the network).
        """
        neuron = IFneuron.IFneuron(neuron_id)
        if neuron_id in self.all_neurons:
            print(f'Neuron: {neuron_id} already exists. Use a different id.')
            return
        self.all_neurons[neuron_id] = neuron

    def add_neuron_connection(self, presynaptic_neuron_id: str, postsynaptic_neuron_id: str, weight: float):
        """
        Add a directed connection: presynaptic -> postsynaptic with given weight.
        Weight > 0 is excitatory; weight < 0 is inhibitory.
        """
        connection = (self.all_neurons[presynaptic_neuron_id], weight)
        self.all_neurons[postsynaptic_neuron_id].receptors.append(connection)

    def set_direct_stimulation_time_ms(self, neuron_id: str, stimulation_time_ms: list):
        """
        Set absolute times (in ms within the *current* run) when this neuron should fire.
        For in-domain trials, we typically set zero or one time per trial for A/B.
        """
        times_sorted = sorted(list(stimulation_time_ms))
        n = self.all_neurons[neuron_id]
        # queue that will be popped during simulation:
        n.t_directstim_ms = list(times_sorted)
        # NEW: immutable copy preserved for writing:
        n.t_directstim_ms_orig = list(times_sorted)

    # -------- Running the simulation --------
    def run_simulation(self, run_time_ms: int, record_membrane_potential: bool):
        """
        Run for 'run_time_ms' steps at 1 ms resolution.
        Each neuron updates once per ms step.
        """
        self._run_time_len = int(run_time_ms)
        for step in range(self._run_time_len):
            for n in self.all_neurons.values():
                n.update(step, record_membrane_potential)

    # -------- Accessors for analysis/writer --------
    def get_neuron_spike_times_ms(self, neuron_id: str) -> list:
        """Return a list of spike times (ms) for the given neuron."""
        return self.all_neurons[neuron_id].t_act_ms

    def get_neuron_spike_train(self, neuron_id: str) -> np.ndarray:
        """
        Return a binary vector (length = run_time_ms) with 1s at spike indices.
        If the last run length is unknown, infer length from recorded time vector.
        """
        T = self._run_time_len if self._run_time_len > 0 else len(self.get_neuron_run_time_ms(neuron_id))
        spike_train = np.zeros(T, dtype=int)
        for t in self.get_neuron_spike_times_ms(neuron_id):
            if 0 <= int(t) < T:
                spike_train[int(t)] = 1
        return spike_train

    def get_neuron_membrane_potentials(self, neuron_id: str) -> list:
        """Return Vm trace recorded for the given neuron."""
        return self.all_neurons[neuron_id].Vm_recorded

    def get_neuron_run_time_ms(self, neuron_id: str) -> list:
        """Return time vector (ms) corresponding to the Vm trace."""
        return self.all_neurons[neuron_id].t_recorded_ms

    def get_neuron_receptors(self, neuron_id: str) -> list:
        """Return list of incoming (source_id, weight) for inspection/debug."""
        receptors = []
        for connection in self.all_neurons[neuron_id].receptors:
            receptors.append((connection[0].id, connection[1]))
        return receptors

    def get_all_neurons(self) -> list:
        """Return list of IFneuron objects."""
        return list(self.all_neurons.values())

    def get_all_neuron_ids(self) -> list:
        """Return list of neuron ids."""
        return list(self.all_neurons.keys())

    def get_network_id(self) -> str:
        """Return network id label."""
        return self.id

    def print_all_neurons(self):
        """Debug print of internal neuron dict."""
        print(self.all_neurons)

    # -------- Stimulus helpers used by writer --------
    def get_neuron_stim_times(self, neuron_id: str) -> list:
        """Return a (copy of) the absolute stim times list for a neuron."""
        n = self.all_neurons[neuron_id]
        # Prefer the immutable copy if present:
        src = n.t_directstim_ms_orig if hasattr(n, "t_directstim_ms_orig") and n.t_directstim_ms_orig else n.t_directstim_ms
        return list(src)

    def get_neuron_stim_vector(self, neuron_id: str) -> np.ndarray:
        """
        Return a binary vector of length run_time with 1 at each stim time.
        Useful for GT writer to include A_stim/B_stim columns.
        """
        T = self._run_time_len if self._run_time_len > 0 else len(self.get_neuron_run_time_ms(neuron_id))
        v = np.zeros(T, dtype=int)
        for t in self.get_neuron_stim_times(neuron_id):
            tt = int(t)
            if 0 <= tt < T:
                v[tt] = 1
        return v
