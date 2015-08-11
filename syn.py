#!/usr/bin/env python
# -*- coding: utf-8 -*-
from brian2 import *
from numpy.random import choice

def SynParams(
    w_exc = 1.0,
    w_inh = -1.0,
    tau_exc = 10*ms,
    tau_inh = 100*ms,
    delay = 1*ms,
    condition = 'i != j',
    connectivity = 0.1):
    return {'w_exc': w_exc, 'w_inh': w_inh, 'tau_exc': tau_exc, 'tau_inh': tau_inh,
            'delay': delay, 'condition': condition, 'connectivity': connectivity}

def ExcitatorySynapses(neurons, params=SynParams(), dt=None):
    return Synapses(neurons, dt=dt, namespace=params,
        model="""
        w : 1
        dge_syn/dt = -ge_syn / tau_exc : 1
        ge_post = ge_syn : 1 (summed)
        """, pre="ge_syn = w", post="ge_syn = 0")

def InhibitorySynapses(neurons, params=SynParams(), dt=None):
    return Synapses(neurons, dt=dt, namespace=params,
        model="""
        w : 1
        dgi_syn/dt = -gi_syn / tau_inh : 1
        gi_post = gi_syn : 1 (summed)
        """, pre="gi_syn = w", post="gi_syn = 0")

def ConnectSparse(synapses, condition, connectivity, initial_w, delay):
    synapses.connect(condition, p=connectivity)
    synapses.w = initial_w
    synapses.delay = delay

if __name__ == "__main__":
    from lif import *
    
    prefs.codegen.target = 'numpy'
    defaultclock.dt = 1*ms
    duration = 1*second
    
    figure()
    
    params = LifParams(constant_input=0.8)
    params.update(SynParams())
    neurons = LifNeurons(2, params)
    synapses = ExcitatorySynapses(neurons, params)
    synapses.connect(0, 1)
    synapses.w = 2.0
    spike_monitor = SpikeMonitor(neurons)
    state_monitor = StateMonitor(neurons, 'v', record = True, when = 'thresholds')
    network = Network()
    network.add(neurons, synapses, spike_monitor, state_monitor)
    network.run(duration, report='stdout', namespace={})
    
    subplot(211)
    title("Excitatory Synapse Induces Postsynaptic Spikes")
    v = state_monitor.v[:]
    for spike in zip(spike_monitor.i, spike_monitor.t):
        v[spike[0], int(spike[1] / defaultclock.dt)] = params['eq_exc']
    plot(state_monitor.t/ms, v[0], label = 'v0')
    plot(state_monitor.t/ms, v[1], label = 'v1')
    legend()
    
    params = LifParams(constant_input=0.8)
    params.update(SynParams())
    neurons = LifNeurons(2, params)
    synapses = InhibitorySynapses(neurons, params)
    synapses.connect(0, 1)
    synapses.w = -2.0
    spike_monitor = SpikeMonitor(neurons)
    state_monitor = StateMonitor(neurons, 'v', record=True, when='thresholds')
    network = Network()
    network.add(neurons, synapses, spike_monitor, state_monitor)
    network.run(duration, report='stdout', namespace={})
    
    subplot(212)
    title("Inhibitory Synapse Prevents Postsynaptic Spikes")
    v = state_monitor.v[:]
    for spike in zip(spike_monitor.i, spike_monitor.t):
        v[spike[0], int(spike[1] / defaultclock.dt)] = params['eq_exc']
    plot(state_monitor.t/ms, v[0], label = 'v0')
    plot(state_monitor.t/ms, v[1], label = 'v1')
    legend()
    
    show()
