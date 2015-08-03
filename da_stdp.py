# -*- coding: utf-8 -*-
from brian2 import *

def DaStdpParams(
    tau_pos = 20*ms,
    tau_neg = 20*ms,
    tau_elig = 1000*ms,
    a_pos = 1.0,
    a_neg = -1.5,
    w_exc = 0.5,
    w_min = 0.0,
    w_max = 1.0,
    post_delay = 1*ms,
    condition = 'i != j',
    connectivity = 0.1):
    return {'tau_pos': tau_pos, 'tau_neg': tau_neg, 'tau_elig': tau_elig,
            'a_pos': a_pos, 'a_neg': a_neg, 'w_exc': w_exc, 'w_min': w_min, 'w_max': w_max,
            'post_delay': post_delay, 'condition': condition, 'connectivity': connectivity}

def DaStdpSynapses(source, target=None, params=DaStdpParams(), dt=None):
    update_model = """
    r : 1 (shared)
    dl/dt = -l / tau_elig : 1
    dw/dt = l * r * int(w >= w_min) * int(w <= w_max) : 1
    dge/dt = -ge / tau_exc : 1
    ge_total_post = ge : 1 (summed)
    dstdp_pos/dt = -stdp_pos / tau_pos : 1 (event-driven)
    dstdp_neg/dt = -stdp_neg / tau_neg : 1 (event-driven)
    """
    pre_model = """
    stdp_pos += a_pos
    l += stdp_neg
    ge = w
    """
    post_model = """
    stdp_neg += a_neg
    l += stdp_pos
    ge = 0
    """
    synapses = Synapses(source, target=target, model=update_model, pre=pre_model, post=post_model,
                 delay={'post': params['post_delay']}, dt=dt, namespace=params)
    synapses.connect(params['condition'], p=params['connectivity'])
    synapses.w = params['w_exc']
    return synapses

def RewardUnit(synapses, reward_function, dt=None):
    @network_operation(dt=dt)
    def reward_unit():
        synapses.r = reward_function(synapses)
    return reward_unit

if __name__ == "__main__":
    from scipy import signal, stats
    from lif import *
    from syn import *
    
    prefs.codegen.target = 'weave'
    defaultclock.dt = 1*ms
    
    duration = 10*second
    binsize = 1.0*second
    n = 1000
    ne = 800
    
    params = LifParams()
    params.update(SynParams())
    params.update(DaStdpParams())
    neurons = LifNeurons(n, params)
    excitatory_synapses = DaStdpSynapses(neurons[:ne], neurons, params)
    excitatory_synapses.w = 0.5
    inhibitory_synapses = InhibitorySynapses(neurons[ne:], neurons, params)
    inhibitory_synapses.w = -1.0
    def reward_function(synapses):
        return synapses.r * 0.9 + (0.01 if int(synapses.t/ms) % 2000 == 1000 else 0)
    reward_unit = RewardUnit(excitatory_synapses, reward_function, 10*ms)
    
    state_monitor = StateMonitor(excitatory_synapses, ('w', 'l', 'r'), range(10))
    
    network = Network()
    network.add(neurons, excitatory_synapses, inhibitory_synapses, reward_unit, state_monitor)
    network.run(duration, report='stdout', report_period=binsize, namespace={})
    
    figure()
    subplot(211)
    title("Synaptic Weights")
    for i in range(10):
        plot(state_monitor.t/second, state_monitor.w[i], label="Synapse {0}".format(i))
    xlabel('Time (s)')
    ylabel('Weight')
    subplot(212)
    plot(state_monitor.t/second, state_monitor.r[0], label="Reward")
    xlabel('Time (s)')
    ylabel('Reward')
    show()
