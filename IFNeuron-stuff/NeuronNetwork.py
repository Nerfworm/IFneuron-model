import IFneuron

class NeuronNetwork:
    '''
    Data structure for storing networks of IFneurons.
    '''
    def __init__(self, id:'str'):
        self.id = id
        self.all_neurons = {}

    def add_neuron(self, neuron_id:str):
        neuron = IFneuron(neuron_id)
        if neuron in self.all_neurons:
            print(f'Neuron: {neuron.id} already exists. Use a different id.')
            return
        self.all_neurons[neuron_id] = neuron
    
    def add_neuron_connection(self, presynaptic_neuron_id:str, postsynaptic_neuron_id:str, weight:float):
        '''
        presynaptic -> postsynaptic.
        presynaptic_neuron_id is the 'sending' neuron.
        postsynaptic_neuron_id is the 'receiving' neuron.
        '''
        connection = (self.all_neurons[presynaptic_neuron_id], weight)
        self.all_neurons[postsynaptic_neuron_id].receptors.append(connection)
        # print(f'Connection formed: {presynaptic_neuron_id} -> {postsynaptic_neuron_id}')  # If you want to see whats happening

    def set_direct_stimulation_time_ms(self, neuron_id:str, stimulation_time_ms:list):
        self.all_neurons[neuron_id].t_directstim_ms = stimulation_time_ms
        # print(f'Set neuron: {neuron_id} stim times(ms) to: {self.all_neurons[neuron_id].t_directstim_ms}')

    def run_simulation(self, run_time_ms:float, record_membrane_potential:bool):
        for step in range(run_time_ms):
            for i in self.all_neurons.values():
                i.update(step, record_membrane_potential)

    def get_neuron_spike_times_ms(self, neuron_id:str) -> list:
        '''
        Returns individual neuron spike times(ms) as a list.
        '''
        return self.all_neurons[neuron_id].t_act_ms
    
    def get_neuron_membrane_potentials(self, neuron_id:str) -> list:
        '''
        Returns individual neuron membrane potentials as a list.
        '''
        return self.all_neurons[neuron_id].Vm_recorded
    
    def get_neuron_run_time_ms(self, neuron_id:str) -> list:
        '''
        Returns individual neuron run time(ms) as a list.
        This can be used to get the run time(ms) of the system.
        '''
        return self.all_neurons[neuron_id].t_recorded_ms

    def get_all_neurons(self) -> list:
        '''
        Returns list of all neuron objects
        '''
        return list(self.all_neurons.values())
    
    def get_all_neuron_ids(self) -> list:
        '''
        Returns list of all neuron ids
        '''
        return list(self.all_neurons.keys())

    def print_all_neurons(self):
        print(self.all_neurons)