# IFneuron.py
# ----------------------------
# Minimal integrate-and-fire neuron used to build tiny circuits.
# This class supports:
#   • Direct stimulation (forcing a spike at specific ms times)
#   • Synaptic input from other neurons (double-exponential PSPs)
#   • Simple after-hyperpolarization (AHP)
#   • Optional spontaneous activity (disabled by default)
#
# For our in-domain trial generator, each trial runs for a fixed number
# of milliseconds (e.g., 100 ms). Neurons A and/or B may receive ONE
# direct stimulus time inside that trial window, or none (00 case).
# The rest of the circuit (C, D, E) responds via synaptic connections.

import numpy as np
import scipy.stats as stats

def dblexp(amp: float, tau_rise: float, tau_decay: float, tdiff: float) -> float:
    """
    Double-exponential postsynaptic potential kernel.
    Returns 0 if the event has not happened yet (tdiff < 0).
    """
    if tdiff < 0:
        return 0.0
    return amp * (-np.exp(-tdiff / tau_rise) + np.exp(-tdiff / tau_decay))

class IFneuron:
    """
    Simple integrate-and-fire neuron with:
      - Direct stim times (absolute ms within the current run)
      - Synaptic inputs from other neurons (each with a weight)
      - Threshold crossing to register spikes
      - Basic refractory-like behavior using a spike term + AHP
    """
    def __init__(self, id: str):
        # Identity / bookkeeping
        self.id = id

        # Direct stimulation: when to force spikes (absolute times in ms)
        self.t_directstim_ms = []
        self.t_directstim_ms_orig = []

        # Membrane potential state and parameters (mV)
        self.Vm_mV = -60.0            # current membrane potential
        self.Vrest_mV = -60.0         # resting potential (baseline)
        self.Vact_mV = -50.0          # threshold for spike detection

        # After-hyperpolarization (AHP) effect
        self.Vahp_mV = -20.0          # AHP amplitude (negative drives Vm down)
        self.tau_AHP_ms = 30.0        # AHP decay time constant

        # Synaptic PSP parameters (identical for all receptors here)
        self.tau_PSPr = 5.0           # PSP rise time constant
        self.tau_PSPd = 25.0          # PSP decay time constant
        self.vPSP = 20.0              # PSP amplitude scale

        # Optional spontaneous activity (disabled when mean=0)
        self.tau_spont_mean_stdev_ms = (0, 0)
        self.t_spont_next = -1
        self.dt_spont_dist = None

        # Incoming synapses: list of (source_neuron, weight)
        self.receptors = []

        # Internal step state
        self.t_ms = 0                 # last update time (ms)
        self._has_spiked = False
        self.in_absref = False        # pseudo absolute refractory flag
        self.t_act_ms = []            # spike times (ms)
        self._dt_act_ms = None        # ms since last spike (for PSP/AHP)

        # Recording buffers (time series over the run)
        self.t_recorded_ms = []       # times recorded
        self.Vm_recorded = []         # Vm values recorded

    # ------------ Setup helpers ------------
    def attach_direct_stim(self, t_ms: float):
        """Append a single direct stimulation time (ms) for this run."""
        self.t_directstim_ms.append(t_ms)

    def set_spontaneous_activity(self, mean_stdev: tuple):
        """
        Enable/disable spontaneous activity.
        mean_stdev = (mean_ms, stdev_ms). If mean or stdev is 0, disable.
        """
        self.tau_spont_mean_stdev_ms = mean_stdev
        mu, sigma = self.tau_spont_mean_stdev_ms
        if mu == 0 or sigma == 0:
            self.dt_spont_dist = None
            return
        # Truncated normal over [0, 2*mu]
        a, b = 0, 2 * mu
        self.dt_spont_dist = stats.truncnorm(
            (a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma
        )

    # ------------ Recording helpers ------------
    def record(self, t_ms: float):
        """Record current time and membrane potential."""
        self.t_recorded_ms.append(t_ms)
        self.Vm_recorded.append(self.Vm_mV)

    # ------------ Spike-state helpers ------------
    def has_spiked(self) -> bool:
        """Return True if at least one spike has occurred this run."""
        self._has_spiked = len(self.t_act_ms) > 0
        return self._has_spiked

    def dt_act_ms(self, t_ms: float) -> float:
        """
        Return ms elapsed since last spike. Large sentinel if never spiked.
        Used by PSP/AHP to compute time since last presynaptic spike.
        """
        if self._has_spiked:
            self._dt_act_ms = t_ms - self.t_act_ms[-1]
            return self._dt_act_ms
        return 1e9

    # ------------ Components contributing to Vm ------------
    def vSpike_t(self, t_ms: float) -> float:
        """
        Represents a brief spike pulse effect + sets a pseudo absolute refractory.
        If the last spike was <= 1 ms ago, return a large positive pulse and
        mark in_absref to block immediate re-firing via threshold.
        """
        if not self._has_spiked:
            return 0.0
        self.in_absref = self._dt_act_ms <= 1.0
        if self.in_absref:
            return 60.0  # spike pulse
        return 0.0

    def vAHP_t(self, t_ms: float) -> float:
        """After-hyperpolarization decays from the last spike."""
        if not self._has_spiked:
            return 0.0
        if self.in_absref:
            return 0.0
        return self.Vahp_mV * np.exp(-self._dt_act_ms / self.tau_AHP_ms)

    def vPSP_t(self, t_ms: float) -> float:
        """
        Sum synaptic PSPs from all presynaptic neurons that have spiked.
        Uses double-exponential kernel per presynaptic source.
        """
        vPSPt = 0.0
        for src_cell, weight in self.receptors:
            if src_cell.has_spiked():
                dtPSP = src_cell.dt_act_ms(t_ms)
                vPSPt += dblexp(weight * self.vPSP, self.tau_PSPr, self.tau_PSPd, dtPSP)
        return vPSPt

    # ------------ Update & detection ------------
    def update_Vm(self, t_ms: float, recording: bool):
        """
        Compute Vm(t) = Vrest + vSpike(t) + vAHP(t) + vPSP(t).
        Optionally record Vm and time.
        """
        # Update dt since last spike (for PSP/AHP terms)
        if self.has_spiked():
            self.dt_act_ms(t_ms)

        # Aggregate terms
        vSpike_t = self.vSpike_t(t_ms)
        vAHP_t = self.vAHP_t(t_ms)
        vPSP_t = self.vPSP_t(t_ms)

        # New membrane potential
        self.Vm_mV = self.Vrest_mV + vSpike_t + vAHP_t + vPSP_t

        # Save trace if requested
        if recording:
            self.record(t_ms)

    def detect_threshold(self, t_ms: float):
        """If not in pseudo-absolute-refractory and Vm >= threshold, log a spike."""
        if self.in_absref:
            return
        if self.Vm_mV >= self.Vact_mV:
            self.t_act_ms.append(t_ms)

    def spontaneous_activity(self, t_ms: float):
        """
        Optional spontaneous spiking schedule.
        Disabled by default (mean=0). If enabled, it schedules the next
        spontaneous spike time and triggers spikes accordingly.
        """
        if self.in_absref:
            return
        mu = self.tau_spont_mean_stdev_ms[0]
        if mu == 0:  # disabled
            return
        if t_ms >= self.t_spont_next:
            if self.t_spont_next >= 0:
                self.t_act_ms.append(t_ms)
            dt_spont = float(self.dt_spont_dist.rvs(1)[0])
            self.t_spont_next = t_ms + dt_spont

    def update(self, t_ms: float, recording: bool):
        """
        One simulation step at time t_ms:
          1) Fire direct stim if scheduled at/ before this time
          2) Update Vm, check threshold crossing, optional spontaneous spike
          3) Store last update time
        """
        tdiff_ms = t_ms - self.t_ms
        if tdiff_ms < 0:
            return  # ignore out-of-order updates

        # Direct stimulation: pop the next stim when its time arrives
        if self.t_directstim_ms:
            if self.t_directstim_ms[0] <= t_ms:
                tfire_ms = self.t_directstim_ms.pop(0)
                self.t_act_ms.append(tfire_ms)

        # Update Vm and detect threshold crossing
        self.update_Vm(t_ms, recording)
        self.detect_threshold(t_ms)

        # Optional spontaneous schedule
        self.spontaneous_activity(t_ms)

        # Remember last time
        self.t_ms = t_ms

    def get_recording(self) -> dict:
        """Convenience accessor for recorded Vm."""
        return {'Vm': self.Vm_recorded}
