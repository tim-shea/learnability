# -*- coding: utf-8 -*-
from brian2 import *

def SynParams(
    tau_exc = 10*ms,
    tau_inh = 100*ms):
    return {'tau_exc': tau_exc, 'tau_inh': tau_inh}

def ExcitatorySynapses(N, params=SynParams(), dt=None):
    update_model = """
    w : 1
    dge/dt = -ge / tau_exc : 1
    ge_total_post = ge : 1 (summed)
    """
    pre_model = "ge = w"
    post_model = "ge = 0"
    S = Synapses(N, model=update_model, pre=pre_model, post=post_model,
                 delay={'post': 1*ms}, dt=dt, namespace=params)
    return S

def InhibitorySynapses(N, params=SynParams(), dt=None):
    update_model = """
    w : 1
    dgi/dt = -gi / tau_inh : 1
    gi_total_post = gi : 1 (summed)
    """
    pre_model = "gi = w"
    post_model = "gi = 0"
    S = Synapses(N, model=update_model, pre=pre_model, post=post_model,
                 delay={'post': 1*ms}, dt=dt, namespace=params)
    return S

if __name__ == "__main__":
    from lif import *
    
    prefs.codegen.target = 'weave'
    defaultclock.dt = 1*ms
    duration = 1*second
    
    figure()
    
    params = LifParams()
    params.update(SynParams())
    N = LifNeurons(2, params)
    S = ExcitatorySynapses(N, params)
    S.connect(0, 1)
    S.w = 2.0
    spike_monitor = SpikeMonitor(N)
    state_monitor = StateMonitor(N, 'v', record = True, when = 'thresholds')
    network = Network()
    network.add(N, S, spike_monitor, state_monitor)
    network.run(duration, report='stdout', namespace={})
    
    subplot(211)
    title("Excitatory Synapse Induces Postsynaptic Spikes")
    v = state_monitor.v[:]
    for spike in zip(spike_monitor.i, spike_monitor.t):
        v[spike[0], int(spike[1] / defaultclock.dt)] = params['v_peak']
    plot(state_monitor.t/ms, v[0], label = 'v0')
    plot(state_monitor.t/ms, v[1], label = 'v1')
    legend()
    
    N = LifNeurons(2, params)
    S = InhibitorySynapses(N, params)
    S.connect(0, 1)
    S.w = -2.0
    spike_monitor = SpikeMonitor(N)
    state_monitor = StateMonitor(N, 'v', record=True, when='thresholds')
    network = Network()
    network.add(N, S, spike_monitor, state_monitor)
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
