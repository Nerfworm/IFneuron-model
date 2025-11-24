# IFneuron.py
# ----------------------------
# Integrate-and-fire neuron model for building simple neural circuits.
#
# Features:
#   • Direct stimulation at specific times
#   • Synaptic input from other neurons (double-exponential PSPs)
#   • After-hyperpolarization (AHP) 
#   • Optional spontaneous activity
#
# Example usage:
#     # Create and connect neurons
#     neuronA = IFneuron("A")
#     neuronB = IFneuron("B")
#     neuronB.receptors.append((neuronA, 1.0))  # B receives input from A
#     
#     # Schedule direct stimulation
#     neuronA.attach_direct_stim(10.0)
#     
#     # Simulate for 100ms
#     for t in range(100):
#         neuronA.update(t, recording=True)
#         neuronB.update(t, recording=True)
#     
#     # Access results
#     spike_times = neuronA.t_act_ms
#     voltage_trace = neuronA.get_recording()

import numpy as np
import scipy.stats as stats
from typing import List, Tuple, Optional, Dict


def dblexp(amp: float, tau_rise: float, tau_decay: float, tdiff: float) -> float:
    """
    Double-exponential postsynaptic potential kernel.
    Standard model in computational neuroscience for PSP/PSC dynamics.
    
    Implements: amp * (-exp(-t/τr) + exp(-t/τd)) for t >= 0
    
    Args:
        amp: Peak amplitude of the PSP (mV)
        tau_rise: Rise time constant (ms) - controls how fast PSP reaches peak
        tau_decay: Decay time constant (ms) - controls how fast PSP returns to baseline
        tdiff: Time since presynaptic spike (ms)
    
    Returns:
        PSP amplitude at time tdiff (returns 0 if tdiff < 0)
    """
    if tdiff < 0:
        return 0.0
    return amp * (-np.exp(-tdiff / tau_rise) + np.exp(-tdiff / tau_decay))


class IFneuron:
    """
    Integrate-and-fire neuron model.
    
    Membrane dynamics:
      Vm(t) = Vrest + vSpike(t) + vAHP(t) + vPSP(t)
    
    Key parameters:
      - Time resolution: 1ms steps
      - Absolute refractory period: 1ms
      - Threshold: -50mV (default)
      - Resting potential: -60mV
    """
    
    def __init__(self, id: str):
        # Identity
        self.id = id

        # Direct stimulation times (ms)
        self.t_directstim_ms: List[float] = []
        self.t_directstim_ms_orig: List[float] = []  # backup

        # Membrane potential parameters (mV)
        self.Vm_mV: float = -60.0            # current membrane potential
        self.Vrest_mV: float = -60.0         # resting potential
        self.Vact_mV: float = -50.0          # spike threshold

        # After-hyperpolarization parameters
        self.Vahp_mV: float = -20.0          # AHP amplitude (negative drives Vm down)
        self.tau_AHP_ms: float = 30.0        # AHP decay time constant

        # Synaptic PSP parameters
        self.tau_PSPr: float = 5.0           # PSP rise time constant
        self.tau_PSPd: float = 25.0          # PSP decay time constant
        self.vPSP: float = 20.0              # PSP amplitude scale

        # Spontaneous activity (disabled when mean=0)
        self.tau_spont_mean_stdev_ms: Tuple[float, float] = (0, 0)
        self.t_spont_next: float = -1
        self.dt_spont_dist: Optional[stats.rv_continuous] = None

        # Synaptic inputs: list of (source_neuron, weight)
        self.receptors: List[Tuple['IFneuron', float]] = []

        # State variables
        self.t_ms: float = 0                 # last update time
        self._has_spiked: bool = False
        self.in_absref: bool = False         # absolute refractory flag
        self.t_act_ms: List[float] = []      # spike times
        self._dt_act_ms: Optional[float] = None

        # Recording buffers
        self.t_recorded_ms: List[float] = []
        self.Vm_recorded: List[float] = []

    def attach_direct_stim(self, t_ms: float):
        """Add a direct stimulation time."""
        self.t_directstim_ms.append(t_ms)

    def set_spontaneous_activity(self, mean_stdev: Tuple[float, float]):
        """
        Configure spontaneous firing.
        
        Args:
            mean_stdev: (mean_ms, stdev_ms) for interspike intervals.
                       Disabled if mean or stdev is 0.
                       Uses truncated normal distribution over [0, 2*mean].
        """
        self.tau_spont_mean_stdev_ms = mean_stdev
        mu, sigma = self.tau_spont_mean_stdev_ms
        if mu == 0 or sigma == 0:
            self.dt_spont_dist = None
            return
        # Truncated normal prevents negative or extreme intervals
        a, b = 0, 2 * mu
        self.dt_spont_dist = stats.truncnorm(
            (a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma
        )

    def record(self, t_ms: float):
        """Record current state for analysis."""
        self.t_recorded_ms.append(t_ms)
        self.Vm_recorded.append(self.Vm_mV)

    def has_spiked(self) -> bool:
        """Check if neuron has fired at least once."""
        self._has_spiked = len(self.t_act_ms) > 0
        return self._has_spiked

    def dt_act_ms(self, t_ms: float) -> float:
        """
        Time since last spike.
        
        Returns:
            ms since last spike, or 1e9 if never spiked.
            Large value ensures PSP/AHP calculations return ~0 for inactive neurons.
        """
        if self._has_spiked:
            self._dt_act_ms = t_ms - self.t_act_ms[-1]
            return self._dt_act_ms
        return 1e9

    def vSpike_t(self, t_ms: float) -> float:
        """
        Spike depolarization component.
        Sets absolute refractory flag to prevent immediate re-firing.
        
        Returns:
            60mV during 1ms refractory period (visual spike peak), 0 otherwise
        """
        if not self._has_spiked:
            return 0.0
        self.in_absref = self._dt_act_ms <= 1.0  # 1ms absolute refractory
        if self.in_absref:
            return 60.0
        return 0.0

    def vAHP_t(self, t_ms: float) -> float:
        """
        After-hyperpolarization component.
        Models K+ conductance that hyperpolarizes the cell after firing.
        
        Returns:
            Negative voltage that decays exponentially with tau_AHP_ms
        """
        if not self._has_spiked or self.in_absref:
            return 0.0
        return self.Vahp_mV * np.exp(-self._dt_act_ms / self.tau_AHP_ms)

    def vPSP_t(self, t_ms: float) -> float:
        """
        Sum synaptic inputs from all connected neurons.
        Uses double-exponential kernel for each presynaptic source.
        
        Returns:
            Total PSP contribution (linear summation assumes synaptic independence)
        """
        vPSPt = 0.0
        for src_cell, weight in self.receptors:
            if src_cell.has_spiked():
                dtPSP = src_cell.dt_act_ms(t_ms)
                vPSPt += dblexp(weight * self.vPSP, self.tau_PSPr, self.tau_PSPd, dtPSP)
        return vPSPt

    def update_Vm(self, t_ms: float, recording: bool):
        """
        Update membrane potential: Vm(t) = Vrest + vSpike(t) + vAHP(t) + vPSP(t)
        
        Note: Order matters - vSpike_t() sets the in_absref flag used by other components.
        """
        # Update time since last spike
        if self.has_spiked():
            self.dt_act_ms(t_ms)

        # Calculate voltage components
        vSpike_t = self.vSpike_t(t_ms)
        vAHP_t = self.vAHP_t(t_ms)
        vPSP_t = self.vPSP_t(t_ms)

        # Update membrane potential
        self.Vm_mV = self.Vrest_mV + vSpike_t + vAHP_t + vPSP_t

        if recording:
            self.record(t_ms)

    def detect_threshold(self, t_ms: float):
        """Check for threshold crossing and register spike."""
        if not self.in_absref and self.Vm_mV >= self.Vact_mV:
            self.t_act_ms.append(t_ms)

    def spontaneous_activity(self, t_ms: float):
        """Handle spontaneous spike generation if enabled."""
        if self.in_absref:
            return
        mu = self.tau_spont_mean_stdev_ms[0]
        if mu == 0:  # Disabled
            return
        if t_ms >= self.t_spont_next:
            if self.t_spont_next >= 0:
                self.t_act_ms.append(t_ms)
            # Schedule next spontaneous spike
            dt_spont = float(self.dt_spont_dist.rvs(1)[0])
            self.t_spont_next = t_ms + dt_spont

    def update(self, t_ms: float, recording: bool = False):
        """
        Main simulation step:
        1. Process direct stimulation
        2. Update membrane potential
        3. Check for threshold crossing
        4. Handle spontaneous activity
        
        Args:
            t_ms: Current simulation time (ms)
            recording: Whether to record Vm trace
        """
        tdiff_ms = t_ms - self.t_ms
        if tdiff_ms < 0:
            return

        # Fire if direct stimulation is scheduled
        if self.t_directstim_ms and self.t_directstim_ms[0] <= t_ms:
            tfire_ms = self.t_directstim_ms.pop(0)
            self.t_act_ms.append(tfire_ms)

        # Update state
        self.update_Vm(t_ms, recording)
        self.detect_threshold(t_ms)
        self.spontaneous_activity(t_ms)

        self.t_ms = t_ms

    def get_recording(self) -> Dict[str, List[float]]:
        """
        Get recorded membrane potential trace.
        
        Returns:
            Dictionary with 'Vm' key containing voltage values
        """
        return {'Vm': self.Vm_recorded}