#!/usr/bin/env python
# -*- coding: utf-8 -*-
from brian2 import *

def LifParams(
    tau_mem = 100*ms,
    v_threshold = 0.75,
    v_reset = 0.0,
    eq_leak = 0.0,
    eq_exc = 1.0,
    eq_inh = -1.0,
    refractory_period = 1*ms,
    constant_input = 0.75,
    random_input = 0.75,
    tau_da = 10*ms):
    return {'v_threshold': v_threshold, 'v_reset': v_reset,
            'refractory_period': refractory_period, 'tau_mem': tau_mem,
            'eq_leak': eq_leak, 'eq_exc': eq_exc, 'eq_inh': eq_inh,
            'constant_input': constant_input, 'random_input': random_input,
            'tau_da': tau_da}

def LifNeurons(n, params=LifParams(), dt=None):
    model="""
    dda/dt = -da / tau_da : 1
    input : 1
    ge : 1
    gi : 1
    I = input + random_input * (-0.5 + rand()) : 1
    g = -(v - eq_leak) + ge * (eq_exc - v) + gi * (v - eq_inh) : 1
    dv/dt = (g + I) / tau_mem : 1 (unless refractory)
    """
    neurons = NeuronGroup(n, model=model, threshold='v > v_threshold', reset="v = v_reset",
                          refractory='refractory_period', namespace=params, dt=dt)
    neurons.v = "v_reset + (v_threshold - v_reset) * rand()"
    neurons.ge = 0
    neurons.gi = 0
    neurons.input = 'constant_input'
    neurons.da = 0
    return neurons

if __name__ == "__main__":
    from matplotlib.pyplot import *
    prefs.codegen.target = 'numpy'
    defaultclock.dt = 1*ms
    def run_lif(duration, constant_input, random_input):
        params = LifParams(constant_input=constant_input, random_input=random_input)
        neurons = LifNeurons(1, params)
        spike_monitor = SpikeMonitor(neurons)
        state_monitor = StateMonitor(neurons, ('v', 'I'), record=True, when='thresholds')
        network = Network()
        network.add(neurons, spike_monitor, state_monitor)
        network.run(duration, namespace={})
        v = state_monitor.v[0]
        for t_spike in spike_monitor.t:
            i_spike = int(t_spike / defaultclock.dt)
            v[i_spike] = params['eq_exc']
        return state_monitor.t/second, v, state_monitor.I[0], size(spike_monitor.i)
    
    figure()
    t, v, I, _ = run_lif(1*second, 0.75, 0.75)
    subplot(121)
    plot(t, I)
    xlabel('t (s)')
    ylabel('I')
    subplot(122)
    plot(t, v)
    xlabel('t (s)')
    ylabel('v')
    show()
    
    figure()
    for i in arange(0.5, 1.0, 0.01):
        _, _, _, spks = run_lif(10*second, i, 0.75)
        plot(i, spks / 10.0, 'o')
    xlabel('Input Scale')
    ylabel('Firing Rate (Hz)')
    show()
