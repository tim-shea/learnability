#!/usr/bin/env python
# -*- coding: utf-8 -*-
from brian2 import *

def LifParams(
    tau_mem = 50*ms,
    v_threshold = 1,
    v_peak = 1.5,
    v_reset = 0,
    eq_leak = 0,
    eq_exc = 1.5,
    eq_inh = 0,
    refractory_period = 5*ms,
    noise_scale = 0.05):
    return {'tau_mem': tau_mem, 'v_threshold': v_threshold, 'v_peak': v_peak, 'v_reset': v_reset,
            'eq_leak': eq_leak, 'eq_exc': eq_exc, 'eq_inh': eq_inh, 'refractory_period': refractory_period,
            'noise_scale': noise_scale}

def LifNeurons(n, params=LifParams(), dt=None):
    neuron_model = """
    ge_total : 1
    gi_total : 1
    dnoise/dt = (-noise + rand())/dt : 1
    I = (eq_exc - v_reset) * tau_mem/ms * noise_scale * noise : 1
    dv/dt = (-(v - eq_leak) + (eq_exc - v) * ge_total + (v - eq_inh) * gi_total + I) / tau_mem : 1 (unless refractory)
    """
    reset_model = "v = v_reset"
    neurons = NeuronGroup(n, neuron_model, threshold='v > v_threshold', reset=reset_model, refractory='refractory_period',
                    namespace=params, dt=dt)
    neurons.v = "v_reset + (v_threshold - v_reset) * rand()"
    return neurons

if __name__ == "__main__":
    from matplotlib.pyplot import *
    prefs.codegen.target = 'numpy'
    defaultclock.dt = 1*ms
    duration = 1*second
    
    params = LifParams()
    neurons = LifNeurons(1, params)
    spike_monitor = SpikeMonitor(neurons)
    state_monitor = StateMonitor(neurons, ('v', 'I'), record=True, when='thresholds')
    network = Network()
    network.add(neurons, spike_monitor, state_monitor)
    network.run(duration, report='stdout', namespace={})
    
    figure()
    plot(state_monitor.t/second, 0.1 * state_monitor.I[0])
    v = state_monitor.v[0]
    for t_spike in spike_monitor.t:
        i_spike = int(t_spike / defaultclock.dt)
        v[i_spike] = params['v_peak']
    plot(state_monitor.t/second, v)
    xlabel('Time (s)')
    ylabel('Input Current')
    show()
