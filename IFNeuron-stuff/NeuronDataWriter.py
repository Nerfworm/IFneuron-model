from NeuronNetwork import *
import numpy as np
import h5py

class NeuronDataWriter:
    def __init__(self, file_name: str = "unlabeled.hdf5"):
        self.file_name = file_name

    def write_trial(self, neuron_network: NeuronNetwork, trial_idx: int):
        """
        Writes a single trial's neuron data to an HDF5 file as a new group.
        Each group contains a 2D dataset: rows=timesteps, columns=features.
        """
        neuron_ids = neuron_network.get_all_neuron_ids()
        time_vector = np.array(neuron_network.get_neuron_run_time_ms(neuron_ids[0]))

        # Build columns and data matrix
        columns = ['time']
        data_matrix = [time_vector]
        for nid in neuron_ids:
            columns.append(f"{nid}_spike_train")
            data_matrix.append(np.array(neuron_network.get_neuron_spike_train(nid)))
            columns.append(f"{nid}_membrane_potential")
            data_matrix.append(np.array(neuron_network.get_neuron_membrane_potentials(nid)))
            # Excluding input for now because we can see what neurons are stimulated.
            # columns.append(f"{nid}_input")
            # data_matrix.append(np.array(neuron_network.get_neuron_input(nid)))

        data_matrix = np.stack(data_matrix, axis=1)  # shape: (timesteps, features)

        with h5py.File(self.file_name, "a") as f:
            group = f.create_group(f"trial_{trial_idx}")
            dset = group.create_dataset("data", data=data_matrix)
            dset.attrs['columns'] = np.array(columns, dtype=h5py.string_dtype(encoding="utf-8"))