# NeuronDataWriter.py
# --------------------------------
# HDF5 writer for neural network simulation trials.
#
# File structure:
#   /trial_{idx}/data           -> (timesteps x features) matrix
#   /trial_{idx}/attrs/columns  -> column names
#   /trial_{idx}/attrs/trial_meta -> JSON metadata
#
# Columns per trial:
#   time                        -> 0..trial_length_ms-1
#   A_stim, B_stim (optional)   -> binary stimulus vectors
#   {Neuron}_spike_train        -> binary spike data
#   {Neuron}_membrane_potential -> continuous Vm traces (mV)
#
# Example usage:
#     writer = NeuronDataWriter("simulation_results.hdf5")
#     
#     # Write trials
#     for trial_idx, (a_bit, b_bit, a_time, b_time) in enumerate(trials):
#         net = build_network()
#         net.set_direct_stimulation_time_ms("Neuron_A", [a_time] if a_time else [])
#         net.set_direct_stimulation_time_ms("Neuron_B", [b_time] if b_time else [])
#         net.run_simulation(100, record_membrane_potential=True)
#         
#         writer.write_trial(
#             net, trial_idx, 100,
#             a_bit, b_bit, a_time, b_time
#         )
#     
#     # Read data
#     with h5py.File("simulation_results.hdf5", "r") as f:
#         data = f["trial_0/data"][:]
#         columns = f["trial_0/data"].attrs["columns"]
#         meta = json.loads(f["trial_0"].attrs["trial_meta"])

from NeuronNetwork import NeuronNetwork
import numpy as np
import h5py
import json
from typing import Optional, List, Dict, Any


class NeuronDataWriter:
    """
    HDF5 writer for neural network simulation trials.
    
    Each trial is stored as a separate group with:
      - Time-series data matrix (timesteps x features)
      - Column names and metadata as attributes
    """
    
    def __init__(self, file_name: str = "unlabeled.hdf5"):
        """
        Initialize writer with target HDF5 file.
        
        Args:
            file_name: Path to HDF5 file (created if doesn't exist)
        """
        self.file_name = file_name

    def write_trial(
        self,
        neuron_network: NeuronNetwork,
        trial_idx: int,
        trial_length_ms: int,
        a_bit: int,
        b_bit: int,
        a_stim_time: Optional[int] = None,
        b_stim_time: Optional[int] = None,
        trial_uid: Optional[str] = None,
    ) -> None:
        """
        Write one simulation trial to HDF5.

        Args:
            neuron_network: Simulated network containing results
            trial_idx: Unique trial index
            trial_length_ms: Trial duration (ms)
            a_bit: Input A bit value (0 or 1)
            b_bit: Input B bit value (0 or 1)
            a_stim_time: When A was stimulated (ms), None if no stim
            b_stim_time: When B was stimulated (ms), None if no stim
            trial_uid: Optional unique identifier string
            
        Raises:
            ValueError: If network empty or invalid parameters
            OSError: If file operations fail
        """
        # Validate inputs
        neuron_ids = neuron_network.get_all_neuron_ids()
        if not neuron_ids:
            raise ValueError("Neuron network has no neurons")
        
        if a_bit not in (0, 1) or b_bit not in (0, 1):
            raise ValueError(f"Bits must be 0 or 1, got a_bit={a_bit}, b_bit={b_bit}")
        
        if a_stim_time is not None and not (0 <= a_stim_time < trial_length_ms):
            raise ValueError(f"a_stim_time={a_stim_time} outside bounds [0, {trial_length_ms})")
        if b_stim_time is not None and not (0 <= b_stim_time < trial_length_ms):
            raise ValueError(f"b_stim_time={b_stim_time} outside bounds [0, {trial_length_ms})")

        # Build data matrix
        time_vector = np.arange(trial_length_ms, dtype=int)
        columns: List[str] = ['time']
        data_cols: List[np.ndarray] = [time_vector]

        # Add stimulus vectors if input neurons exist
        if 'Neuron_A' in neuron_ids:
            columns.append('A_stim')
            data_cols.append(neuron_network.get_neuron_stim_vector('Neuron_A'))
        if 'Neuron_B' in neuron_ids:
            columns.append('B_stim')
            data_cols.append(neuron_network.get_neuron_stim_vector('Neuron_B'))

        # Add spike and membrane potential data for each neuron
        for nid in neuron_ids:
            # Spike train
            columns.append(f"{nid}_spike_train")
            spike_train = neuron_network.get_neuron_spike_train(nid)
            data_cols.append(spike_train)

            # Membrane potential
            columns.append(f"{nid}_membrane_potential")
            vm_trace = neuron_network.get_neuron_membrane_potentials(nid)
            # Pad if recording incomplete
            if len(vm_trace) < trial_length_ms:
                vm_trace = np.pad(vm_trace, (0, trial_length_ms - len(vm_trace)), 
                                 constant_values=0.0)
            data_cols.append(np.array(vm_trace, dtype=float))

        # Stack into matrix: (timesteps, features)
        data_matrix = np.stack(data_cols, axis=1)

        # Write to HDF5
        try:
            with h5py.File(self.file_name, "a") as f:
                group_name = f"trial_{trial_idx}"
                if group_name in f:
                    print(f"Warning: Overwriting existing group {group_name}")
                    del f[group_name]
                
                # Create trial group
                group = f.create_group(group_name)

                # Store data with compression
                dset = group.create_dataset(
                    "data", 
                    data=data_matrix,
                    compression="gzip",
                    compression_opts=4
                )

                # Store column names
                dset.attrs['columns'] = np.array(columns, dtype=h5py.string_dtype(encoding="utf-8"))

                # Store metadata
                meta: Dict[str, Any] = {
                    "trial_index": int(trial_idx),
                    "trial_uid": None if trial_uid is None else str(trial_uid),
                    "trial_length_ms": int(trial_length_ms),
                    "a_bit": int(a_bit),
                    "b_bit": int(b_bit),
                    "a_stim_time": None if a_stim_time is None else int(a_stim_time),
                    "b_stim_time": None if b_stim_time is None else int(b_stim_time),
                    "neuron_ids": neuron_ids,
                    "n_neurons": len(neuron_ids),
                    "n_features": data_matrix.shape[1],
                }
                group.attrs['trial_meta'] = json.dumps(meta, indent=2)
                
        except OSError as e:
            raise OSError(f"Failed to write trial {trial_idx} to {self.file_name}: {e}")
            
    def read_trial_metadata(self, trial_idx: int) -> Dict[str, Any]:
        """
        Read metadata for a specific trial.
        
        Args:
            trial_idx: Trial index to read
            
        Returns:
            Dictionary containing trial metadata
            
        Raises:
            KeyError: If trial doesn't exist
        """
        with h5py.File(self.file_name, "r") as f:
            group_name = f"trial_{trial_idx}"
            if group_name not in f:
                raise KeyError(f"Trial {trial_idx} not found in {self.file_name}")
            return json.loads(f[group_name].attrs['trial_meta'])