# NeuronNetwork.py
# ---------------------------------
# Container for managing networks of IFneurons with simulation and analysis tools.
#
# Features:
#   • Network topology management (neurons and synaptic connections)
#   • Synchronous simulation execution (1ms time steps)
#   • Data extraction (spike trains, membrane potentials, stimulus vectors)
#
# Example usage:
#     # Create a 3-neuron feedforward network
#     net = NeuronNetwork("test_network")
#     
#     # Add neurons
#     net.add_neuron("A")
#     net.add_neuron("B") 
#     net.add_neuron("C")
#     
#     # Wire connections: A->B (excitatory), B->C (excitatory)
#     net.add_neuron_connection("A", "B", weight=1.0)
#     net.add_neuron_connection("B", "C", weight=0.8)
#     
#     # Schedule direct stimulation
#     net.set_direct_stimulation_time_ms("A", [10])
#     
#     # Run simulation
#     net.run_simulation(100, record_membrane_potential=True)
#     
#     # Extract results
#     spike_times = net.get_neuron_spike_times_ms("B")
#     spike_train = net.get_neuron_spike_train("B")
#     vm_trace = net.get_neuron_membrane_potentials("B")

import IFneuron
import numpy as np
from typing import Dict, List, Tuple, Optional


class NeuronNetwork:
    """
    Network container for IFneurons with simulation and analysis tools.
    
    Key properties:
      - Synchronous updates: All neurons update at each 1ms timestep
      - Fixed time resolution: 1ms
      - Direct stimulation: Absolute times (not relative to trial start)
      - Data preservation: Original stimulus times retained after simulation
    """
    
    def __init__(self, id: str):
        """
        Initialize an empty network.
        
        Args:
            id: Unique identifier for this network
        """
        self.id: str = id
        self.all_neurons: Dict[str, IFneuron.IFneuron] = {}
        self._run_time_len: int = 0  # Duration of last simulation

    def add_neuron(self, neuron_id: str) -> None:
        """
        Add a new neuron to the network.
        
        Args:
            neuron_id: Unique identifier for the neuron
        
        Raises:
            ValueError: If neuron ID already exists
        """
        if neuron_id in self.all_neurons:
            raise ValueError(f'Neuron "{neuron_id}" already exists in network')
        
        neuron = IFneuron.IFneuron(neuron_id)
        self.all_neurons[neuron_id] = neuron

    def add_neuron_connection(self, presynaptic_neuron_id: str, 
                            postsynaptic_neuron_id: str, 
                            weight: float) -> None:
        """
        Create a synaptic connection between two neurons.
        
        Args:
            presynaptic_neuron_id: Source neuron (sends signal)
            postsynaptic_neuron_id: Target neuron (receives signal)
            weight: Synaptic weight (positive=excitatory, negative=inhibitory)
        
        Raises:
            KeyError: If either neuron doesn't exist
        """
        if presynaptic_neuron_id not in self.all_neurons:
            raise KeyError(f'Presynaptic neuron "{presynaptic_neuron_id}" not found')
        if postsynaptic_neuron_id not in self.all_neurons:
            raise KeyError(f'Postsynaptic neuron "{postsynaptic_neuron_id}" not found')
            
        connection = (self.all_neurons[presynaptic_neuron_id], weight)
        self.all_neurons[postsynaptic_neuron_id].receptors.append(connection)

    def set_direct_stimulation_time_ms(self, neuron_id: str, 
                                      stimulation_time_ms: List[float]) -> None:
        """
        Schedule direct stimulation times for a neuron.
        
        Direct stimulation forces a spike at the specified times,
        bypassing normal synaptic input. Typically used for input neurons.
        
        Args:
            neuron_id: Target neuron
            stimulation_time_ms: List of absolute times (ms) to stimulate
        
        Raises:
            KeyError: If neuron doesn't exist
        """
        if neuron_id not in self.all_neurons:
            raise KeyError(f'Neuron "{neuron_id}" not found')
            
        times_sorted = sorted(list(stimulation_time_ms))
        n = self.all_neurons[neuron_id]
        n.t_directstim_ms = list(times_sorted)  # Working copy (consumed during simulation)
        n.t_directstim_ms_orig = list(times_sorted)  # Preserved for analysis

    def run_simulation(self, run_time_ms: int, record_membrane_potential: bool) -> None:
        """
        Execute network simulation.
        
        Updates all neurons synchronously at 1ms intervals.
        
        Args:
            run_time_ms: Simulation duration (ms)
            record_membrane_potential: Whether to record Vm traces
        """
        self._run_time_len = int(run_time_ms)
        for step in range(self._run_time_len):
            for n in self.all_neurons.values():
                n.update(step, record_membrane_potential)

    def get_neuron_spike_times_ms(self, neuron_id: str) -> List[float]:
        """
        Get spike times for a neuron.
        
        Returns:
            List of spike times (ms), empty if no spikes
        """
        if neuron_id not in self.all_neurons:
            raise KeyError(f'Neuron "{neuron_id}" not found')
        return self.all_neurons[neuron_id].t_act_ms

    def get_neuron_spike_train(self, neuron_id: str) -> np.ndarray:
        """
        Get binary spike train vector.
        
        Returns:
            Binary array with 1s at spike times, 0s elsewhere
        """
        if neuron_id not in self.all_neurons:
            raise KeyError(f'Neuron "{neuron_id}" not found')
            
        T = self._run_time_len if self._run_time_len > 0 else len(self.get_neuron_run_time_ms(neuron_id))
        spike_train = np.zeros(T, dtype=int)
        
        for t in self.get_neuron_spike_times_ms(neuron_id):
            if 0 <= int(t) < T:
                spike_train[int(t)] = 1
        return spike_train

    def get_neuron_membrane_potentials(self, neuron_id: str) -> List[float]:
        """
        Get recorded membrane potential trace.
        
        Returns:
            List of Vm values (mV), empty if recording disabled
        """
        if neuron_id not in self.all_neurons:
            raise KeyError(f'Neuron "{neuron_id}" not found')
        return self.all_neurons[neuron_id].Vm_recorded

    def get_neuron_run_time_ms(self, neuron_id: str) -> List[float]:
        """
        Get time points when Vm was recorded.
        
        Returns:
            List of recording times (ms)
        """
        if neuron_id not in self.all_neurons:
            raise KeyError(f'Neuron "{neuron_id}" not found')
        return self.all_neurons[neuron_id].t_recorded_ms

    def get_neuron_receptors(self, neuron_id: str) -> List[Tuple[str, float]]:
        """
        Get incoming connections for a neuron.
        
        Returns:
            List of (presynaptic_neuron_id, weight) tuples
        """
        if neuron_id not in self.all_neurons:
            raise KeyError(f'Neuron "{neuron_id}" not found')
            
        receptors = []
        for connection in self.all_neurons[neuron_id].receptors:
            receptors.append((connection[0].id, connection[1]))
        return receptors

    def get_all_neurons(self) -> List[IFneuron.IFneuron]:
        """Get all neuron objects in the network."""
        return list(self.all_neurons.values())

    def get_all_neuron_ids(self) -> List[str]:
        """Get IDs of all neurons."""
        return list(self.all_neurons.keys())

    def get_network_id(self) -> str:
        """Get network identifier."""
        return self.id

    def print_all_neurons(self) -> None:
        """Debug helper: print neuron dictionary."""
        print(f"Network '{self.id}' neurons:", self.all_neurons)

    def get_neuron_stim_times(self, neuron_id: str) -> List[float]:
        """
        Get original stimulus times for a neuron.
        
        Returns the preserved stimulus schedule (not consumed during simulation).
        Essential for ground truth generation.
        
        Returns:
            List of stimulation times (ms)
        """
        if neuron_id not in self.all_neurons:
            raise KeyError(f'Neuron "{neuron_id}" not found')
            
        n = self.all_neurons[neuron_id]
        # Use preserved copy if available
        src = n.t_directstim_ms_orig if hasattr(n, "t_directstim_ms_orig") and n.t_directstim_ms_orig else n.t_directstim_ms
        return list(src)

    def get_neuron_stim_vector(self, neuron_id: str) -> np.ndarray:
        """
        Get binary stimulus vector (1 at each stim time).
        
        For ground truth: represents when stimulation was delivered (input),
        not when neuron spiked (output).
        
        Returns:
            Binary array with 1s at stimulus times
        """
        if neuron_id not in self.all_neurons:
            raise KeyError(f'Neuron "{neuron_id}" not found')
            
        T = self._run_time_len if self._run_time_len > 0 else len(self.get_neuron_run_time_ms(neuron_id))
        v = np.zeros(T, dtype=int)
        
        for t in self.get_neuron_stim_times(neuron_id):
            tt = int(t)
            if 0 <= tt < T:
                v[tt] = 1
        return v