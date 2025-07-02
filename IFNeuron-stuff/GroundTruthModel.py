from NeuronNetwork import *
from utils import generate_random_stimulation_times_ms
import numpy as np
import csv

ground_truth_system = NeuronNetwork('Ground Truth system')
simulation_run_time_ms = 10000       # simulated run time
num_stims = 25                      # number of neuron stimulations
min_time_between_stims_ms = 100     # min amount of time between stimulus
input_stimulus_a = generate_random_stimulation_times_ms(num_stims, simulation_run_time_ms, min_time_between_stims_ms)
input_stimulus_b = generate_random_stimulation_times_ms(num_stims, simulation_run_time_ms, min_time_between_stims_ms)

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

def generate_neuron_data():
    ground_truth_system.run_simulation(simulation_run_time_ms, True)
    # add input times for neurons that apply
    # add noise to the output (not the cells) (maybe record the membrane potentials and add some weird noise to it)
	# Add noise on the input (jiggle stimulus times, and amplitude and pulse to input)
    data = {'NeuronId':[],
            'SpikeTimes':[],
            'MembranePotential':[],
            'ConnectedNeurons':[]}
    for neuron in ground_truth_system.get_all_neurons():
        data['NeuronId'].append(neuron.id)
        data['SpikeTimes'].append(ground_truth_system.get_neuron_spike_times_ms(neuron.id))
        data['MembranePotential'].append(ground_truth_system.get_neuron_membrane_potentials(neuron.id))
        data['ConnectedNeurons'].append(ground_truth_system.get_neuron_receptors(neuron.id))
    
    num_rows = len(data['NeuronId'])
    headers = list(data.keys())
    with open('testData-2.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i in range(num_rows):
            row_data = []
            for header in headers:
                item = data[header][i]
                if isinstance(item, np.floating): # np.floating covers np.float32, np.float64, etc.
                    item = float(item) # Convert to standard Python float
                if isinstance(item, list) or isinstance(item, tuple):
                    # Convert each element within the list/tuple to string
                    # This handles cases like [np.float32(0.1), np.float32(0.2)]
                    processed_list_or_tuple = []
                    for sub_item in item:
                        if isinstance(sub_item, np.floating):
                            processed_list_or_tuple.append(float(sub_item))
                        else:
                            processed_list_or_tuple.append(sub_item)
                    row_data.append(str(processed_list_or_tuple))
                else:
                    row_data.append(item)
            w.writerow(row_data)

generate_neuron_data()