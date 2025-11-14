# NeuronDataWriter.py
# --------------------------------
# Writes each trial as its own group in an HDF5 file.
# Group structure:
#   /trial_{idx}/data           -> (timesteps x features) numeric matrix
#   /trial_{idx}/attrs/columns  -> list of column names (strings)
#   /trial_{idx}/attrs/trial_meta -> JSON string with light metadata
#
# Columns (for each trial):
#   time                      (0..trial_length_ms-1)
#   A_stim (optional)         (binary vector; present if Neuron_A exists)
#   B_stim (optional)         (binary vector; present if Neuron_B exists)
#   Neuron_A_spike_train      (binary)
#   Neuron_A_membrane_potential (float)
#   ...
#   Neuron_E_spike_train
#   Neuron_E_membrane_potential
#
# This mirrors your original 11 neuron columns, plus A_stim/B_stim as helpers.

from NeuronNetwork import *
import numpy as np
import h5py
import json

class NeuronDataWriter:
    def __init__(self, file_name: str = "unlabeled.hdf5"):
        self.file_name = file_name

    def write_trial(
        self,
        neuron_network: NeuronNetwork,
        trial_idx: int,
        trial_length_ms: int,
        a_bit: int,
        b_bit: int,
        a_stim_time: int | None,
        b_stim_time: int | None,
        trial_uid: str | None = None,
    ):
        """
        Append one trial to the HDF5 file as a new group.

        Args:
          neuron_network : the simulated NeuronNetwork for this trial
          trial_idx      : integer index for this trial's group name
          trial_length_ms: number of ms in this trial (rows)
          a_bit, b_bit   : the intended bits (0/1) for inputs A and B
          a_stim_time    : ms within trial for A's stim (or None for 0)
          b_stim_time    : ms within trial for B's stim (or None for 0)
        """
        neuron_ids = neuron_network.get_all_neuron_ids()
        if not neuron_ids:
            raise ValueError("Neuron network has no neurons to write.")

        # Time vector: 0..trial_length_ms-1 (same for all columns)
        time_vector = np.arange(trial_length_ms, dtype=int)

        # Build column list and a list of column arrays to stack
        columns = ['time']
        data_cols = [time_vector]

        # If the canonical input neurons exist, write the stim vectors
        if 'Neuron_A' in neuron_ids:
            columns.append('A_stim')
            data_cols.append(neuron_network.get_neuron_stim_vector('Neuron_A'))
        if 'Neuron_B' in neuron_ids:
            columns.append('B_stim')
            data_cols.append(neuron_network.get_neuron_stim_vector('Neuron_B'))

        # For each neuron, add spike train and Vm columns
        for nid in neuron_ids:
            columns.append(f"{nid}_spike_train")
            data_cols.append(neuron_network.get_neuron_spike_train(nid))

            columns.append(f"{nid}_membrane_potential")
            data_cols.append(np.array(neuron_network.get_neuron_membrane_potentials(nid), dtype=float))

        # Stack into a 2D matrix: (timesteps, features)
        data_matrix = np.stack(data_cols, axis=1)

        # Create/append to the HDF5 file
        with h5py.File(self.file_name, "a") as f:
            group = f.create_group(f"trial_{trial_idx}")

            # Main dataset
            dset = group.create_dataset("data", data=data_matrix)

            # Column names (UTF-8 string dtype)
            dset.attrs['columns'] = np.array(columns, dtype=h5py.string_dtype(encoding="utf-8"))

            # Lightweight metadata (useful for analysis; submissions need not include this)
            meta = {
                "trial_index": int(trial_idx),
                "trial_uid": None if trial_uid is None else str(trial_uid),
                "trial_length_ms": int(trial_length_ms),
                "a_bit": int(a_bit),
                "b_bit": int(b_bit),
                "a_stim_time": None if a_stim_time is None else int(a_stim_time),
                "b_stim_time": None if b_stim_time is None else int(b_stim_time),
                "neuron_ids": neuron_ids,
            }
            group.attrs['trial_meta'] = json.dumps(meta)
