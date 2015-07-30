# -*- coding: utf-8 -*-
from brian2 import *

def SynParams(
    w_exc = 1.0,
    w_inh = -1.0,
    tau_exc = 10*ms,
    tau_inh = 100*ms,
    post_delay = 1*ms,
    condition = 'i != j',
    connectivity = 0.1):
    return {'w_exc': w_exc, 'w_inh': w_inh, 'tau_exc': tau_exc, 'tau_inh': tau_inh,
            'post_delay': post_delay, 'condition': condition, 'connectivity': connectivity}

def ExcitatorySynapses(source, target=None, params=SynParams(), dt=None):
    update_model = """
    w : 1
    dge/dt = -ge / tau_exc : 1
    ge_total_post = ge : 1 (summed)
    """
    pre_model = "ge = w"
    post_model = "ge = 0"
    synapses = Synapses(source, target=target, model=update_model, pre=pre_model, post=post_model,
                        delay={'post': params['post_delay']}, dt=dt, namespace=params)
    synapses.connect(params['condition'], p=params['connectivity'])
    synapses.w = params['w_exc']
    return synapses

def InhibitorySynapses(source, target=None, params=SynParams(), dt=None):
    update_model = """
    w : 1
    dgi/dt = -gi / tau_inh : 1
    gi_total_post = gi : 1 (summed)
    """
    pre_model = "gi = w"
    post_model = "gi = 0"
    synapses = Synapses(source, target=target, model=update_model, pre=pre_model, post=post_model,
                        delay={'post': params['post_delay']}, dt=dt, namespace=params)
    synapses.connect(params['condition'], p=params['connectivity'])
    synapses.w = params['w_inh']
    return synapses

if __name__ == "__main__":
    from lif import *
    
    prefs.codegen.target = 'weave'
    defaultclock.dt = 1*ms
    duration = 1*second
    
    figure()
    
    params = LifParams()
    params.update(SynParams(connectivity=1.0))
    neurons = LifNeurons(2, params)
    synapses = ExcitatorySynapses(neurons[0:1], neurons, params)
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
        v[spike[0], int(spike[1] / defaultclock.dt)] = params['v_peak']
    plot(state_monitor.t/ms, v[0], label = 'v0')
    plot(state_monitor.t/ms, v[1], label = 'v1')
    legend()
    
    neurons = LifNeurons(2, params)
    synapses = InhibitorySynapses(neurons[0:1], neurons, params)
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
        v[spike[0], int(spike[1] / defaultclock.dt)] = params['v_peak']
    plot(state_monitor.t/ms, v[0], label = 'v0')
    plot(state_monitor.t/ms, v[1], label = 'v1')
    legend()
    
    show()
