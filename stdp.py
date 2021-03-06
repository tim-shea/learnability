# -*- coding: utf-8 -*-
from brian2 import *

def StdpParams(
    tau_pos = 20*ms,
    tau_neg = 20*ms,
    r = 0.05,
    a_pos = 1.0,
    a_neg = -1.5,
    w_exc = 0.5,
    w_min = 0.0,
    w_max = 1.0,
    post_delay = 1*ms,
    condition = 'i != j',
    connectivity = 0.1):
    return {'tau_pos': tau_pos, 'tau_neg': tau_neg, 'r': r,
            'a_pos': a_pos, 'a_neg': a_neg, 'w_exc': w_exc, 'w_min': w_min, 'w_max': w_max,
            'post_delay': post_delay, 'condition': condition, 'connectivity': connectivity}

def StdpSynapses(source, target=None, params=StdpParams(), dt=None):
    update_model = """
    w : 1
    dge/dt = -ge / tau_exc : 1
    ge_total_post = ge : 1 (summed)
    dstdp_pos/dt = -stdp_pos / tau_pos : 1
    dstdp_neg/dt = -stdp_neg / tau_neg : 1
    """
    pre_model = """
    stdp_pos += a_pos
    w = clip(w + r * stdp_neg, w_min, w_max)
    ge = w
    """
    post_model = """
    stdp_neg += a_neg
    w = clip(w + r * stdp_pos, w_min, w_max)
    ge = 0
    """
    synapses = Synapses(source, target=target, model=update_model, pre=pre_model, post=post_model,
                 delay={'post': params['post_delay']}, dt=dt, namespace=params)
    synapses.connect(params['condition'], p=params['connectivity'])
    synapses.w = params['w_exc']
    return synapses

if __name__ == "__main__":
    import scipy.stats
    from lif import *
    from syn import *
    prefs.codegen.target = 'weave'
    defaultclock.dt = 1*ms
    params = LifParams(noise_scale=0.025)
    params.update(SynParams())
    params.update(StdpParams(connectivity=1.0))
    neurons = LifNeurons(2, params)
    excitatory_synapses = StdpSynapses(neurons[0:1], neurons, params)
    state_monitor1 = StateMonitor(neurons, 'v', record=True)
    state_monitor2 = StateMonitor(excitatory_synapses, ('stdp_pos', 'stdp_neg'), record=True)
    network = Network()
    network.add(neurons, excitatory_synapses, state_monitor1, state_monitor2)
    network.run(10*second, report='stdout', report_period=10*second)
    
    figure()
    subplot(211)
    title("Membrane Potentials of Two Connected Neurons")
    plot(state_monitor1.t/ms, state_monitor1.v[0]/mV)
    plot(state_monitor1.t/ms, state_monitor1.v[1]/mV)
    subplot(212)
    title("Positive and Negative STDP Traces for Synapse Between Two Neurons")
    plot(state_monitor2.t/ms, state_monitor2.stdp_pos[0])
    plot(state_monitor2.t/ms, state_monitor2.stdp_neg[0])
    show()
    
    periods = 10
    period_duration = 10*second
    total_duration = periods * period_duration
    n = 1000
    ne = 800
    
    params['connectivity'] = 0.1
    neurons = LifNeurons(n, params)
    excitatory_synapses = StdpSynapses(neurons[:ne], neurons, params)
    inhibitory_synapses = InhibitorySynapses(neurons[ne:], neurons, params)
    rate_monitor = PopulationRateMonitor(neurons)
    weights = ndarray((periods + 1, size(excitatory_synapses.w)))
    weights[0] = excitatory_synapses.w
    
    network = Network()
    network.add(neurons, excitatory_synapses, inhibitory_synapses, rate_monitor)
    for period in range(1, periods + 1):
        network.run(period_duration, report='stdout')
        weights[period] = excitatory_synapses.w
    
    subplot(121)
    title("Firing Rate During Simulation")
    rates = scipy.stats.binned_statistic(rate_monitor.t/second, rate_monitor.rate, bins=periods)
    plot(rates[1][1:], rates[0])
    xlabel('Time (s)')
    ylabel('Firing Rate (Hz)')
    
    subplot(122)
    means = mean(weights, axis=1)
    upper = percentile(weights, 75, axis=1)
    lower = percentile(weights, 25, axis=1)
    plot(range(periods + 1), means, '-r')
    plot(range(periods + 1), upper, '-k')
    plot(range(periods + 1), lower, '-k')
    xlabel("Period (10 s)")
    ylabel("Count")
    
    show()
