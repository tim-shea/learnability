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

def DaStdpSynapses(source, params=DaStdpParams(), dt=None):
    synapses = Synapses(source, dt=dt, namespace=params,
        model="""
        dl/dt = -l / tau_elig : 1
        dw/dt = l * da_pre * int(w >= w_min) * int(w <= w_max) : 1
        dge_syn/dt = -ge_syn / tau_exc : 1
        ge_post = ge_syn : 1 (summed)
        dstdp_pos/dt = -stdp_pos / tau_pos : 1 (event-driven)
        dstdp_neg/dt = -stdp_neg / tau_neg : 1 (event-driven)
        """,
        pre="""
        stdp_pos += a_pos
        l += stdp_neg
        ge_syn = w
        """,
        post="""
        stdp_neg += a_neg
        l += stdp_pos
        ge_syn = 0
        """)
    return synapses

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
    
    params = LifParams(constant_input=1.0)
    params.update(SynParams())
    params.update(DaStdpParams())
    neurons = LifNeurons(n, params)
    excitatory_synapses = DaStdpSynapses(neurons, params)
    excitatory_synapses.connect('i != j and i < 800', p=0.1)
    excitatory_synapses.w = 0.5
    inhibitory_synapses = InhibitorySynapses(neurons, params)
    inhibitory_synapses.connect('i != j and i >= 800', p=0.1)
    inhibitory_synapses.w = -1.0
    reward = NeuronGroup(1, "dtimer/dt = 1 : second", threshold="timer > 1*second", reset="timer = 0*second")
    reward_link = Synapses(reward, neurons, pre="da_post += 1", connect=True)
    rate_monitor = PopulationRateMonitor(neurons)
    rew_monitor = StateMonitor(neurons, 'da', record=[0])
    w_monitor = StateMonitor(excitatory_synapses, ('w', 'l'), range(10))
    network = Network()
    network.add(neurons, excitatory_synapses, inhibitory_synapses, reward, reward_link, rate_monitor, rew_monitor, w_monitor)
    network.run(duration, report='stdout', report_period=binsize, namespace={})
    
    figure()
    subplot(311)
    title('Network Firing Rate')
    binned_rate = stats.binned_statistic(rate_monitor.t/second, rate_monitor.rate, bins=100)
    plot(binned_rate[1][:-1], binned_rate[0])
    subplot(312)
    title("Synaptic Weights")
    for i in range(10):
        plot(w_monitor.t/second, w_monitor.w[i], label="Synapse {0}".format(i))
    xlabel('Time (s)')
    ylabel('Weight')
    subplot(313)
    plot(rew_monitor.t/second, rew_monitor.da[0], label="Reward")
    xlabel('Time (s)')
    ylabel('Reward')
    show()
