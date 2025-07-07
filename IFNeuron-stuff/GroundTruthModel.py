from NeuronNetwork import *
from utils import generate_random_stimulation_times_ms
from NeuronDataWriter import NeuronDataWriter
import numpy as np

writer = NeuronDataWriter("test-1000-ms.hdf5")
num_trials = 2                      # number of trials to run
simulation_run_time_ms = 1000       # simulated run time
num_stims = 5                       # number of neuron stimulations
min_time_between_stims_ms = 100     # min amount of time between stimulus
input_stimulus_a = generate_random_stimulation_times_ms(num_stims, simulation_run_time_ms, min_time_between_stims_ms)
input_stimulus_b = generate_random_stimulation_times_ms(num_stims, simulation_run_time_ms, min_time_between_stims_ms)

for trial_idx in range(num_trials):
    ground_truth_system = NeuronNetwork('Ground Truth system')

    ground_truth_system.add_neuron('Neuron_A')  # input neuron
    ground_truth_system.add_neuron('Neuron_B')  # input neuron
    ground_truth_system.add_neuron('Neuron_C')
    ground_truth_system.add_neuron('Neuron_D')
    ground_truth_system.add_neuron('Neuron_E')

    ground_truth_system.add_neuron_connection('Neuron_A', 'Neuron_C', .7)    # Neuron_A -> Neuron_C
    ground_truth_system.add_neuron_connection('Neuron_B', 'Neuron_C', .7)    # Neuron_B -> Neuron_C

    ground_truth_system.add_neuron_connection('Neuron_A', 'Neuron_D', 1.0)   # Neuron_A -> Neuron_D
    ground_truth_system.add_neuron_connection('Neuron_B', 'Neuron_D', 1.0)   # Neuron_B -> Neuron_D

    ground_truth_system.add_neuron_connection('Neuron_D', 'Neuron_E', 1.0)   # Neuron_D -> Neuron_E
    ground_truth_system.add_neuron_connection('Neuron_C', 'Neuron_E', -1.0)  # Neuron_C -> Neuron_E

    ground_truth_system.set_direct_stimulation_time_ms('Neuron_A', input_stimulus_a)

    ground_truth_system.set_direct_stimulation_time_ms('Neuron_B', input_stimulus_b)

    ground_truth_system.run_simulation(simulation_run_time_ms, True)
    writer.write_trial(ground_truth_system, trial_idx)