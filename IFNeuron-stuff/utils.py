import NeuronNetwork
import matplotlib.pyplot
import numpy

def plot_membrane_potential_over_time(simulation:NeuronNetwork):
    '''
    This should be able to plot all the neurons
    in a given NeuronNetwork.
    Needs testing.
    '''
    neuron_ids = simulation.get_all_neuron_ids()
    fig, ax = matplotlib.pyplot.subplots(len(neuron_ids), figsize=(10, len(neuron_ids)*2.5), sharex=True, sharey=True)
    fig.suptitle(f'{simulation.id}: Membrane potential (mV) over Time (ms)')
    fig.supxlabel('Time (ms)')
    fig.supylabel('Membrane Potential (mV)')

    for i, neuron_id in enumerate(simulation.get_all_neuron_ids()):
        ax[i].plot(simulation.get_neuron_run_time_ms(neuron_id),
                simulation.get_neuron_membrane_potentials(neuron_id),
                label='Membrane Potential (mV)')
        ax[i].axhline(simulation.get_all_neurons()[i].Vact_mV,
                    label='Threshold (mV)', linestyle='--', color='r')
        ax[i].set_title(f'{neuron_id}')
        ax[i].legend(loc='upper right')

    matplotlib.pyplot.show

def generate_random_stimulation_times_ms(num_stims:int, max_time_ms:float, min_time_between_stim_ms:float) -> list:
    '''
    Generates ascending stimulus times. Each stim's upper bound scales with its proportion
    of `max_time_ms / num_stims`, ensuring earlier values are statistically denser while
    respecting `min_time_between_stim_ms`.
    '''
    stimulation_times = []
    current_time = 0.0
    time_per_stim_slot = max_time_ms / num_stims

    for i in range(num_stims):
        min_current_stim_time = current_time + min_time_between_stim_ms
        upper_limit_from_distribution = (i + 1) * time_per_stim_slot
        remaining_stims_to_place = num_stims - (i + 1)
        upper_limit_from_remaining_space = max_time_ms - (remaining_stims_to_place * min_time_between_stim_ms)
        upper_bound = min(upper_limit_from_distribution, upper_limit_from_remaining_space)
        lower_bound = max(min_current_stim_time, current_time)

        if lower_bound > upper_bound:
            return stimulation_times

        stim_time = lower_bound if lower_bound == upper_bound else numpy.random.uniform(lower_bound, upper_bound + numpy.finfo(float).eps)
        
        stimulation_times.append(stim_time)
        current_time = stim_time

    return stimulation_times