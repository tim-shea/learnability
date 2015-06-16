# -*- coding: utf-8 -*-
from brian2 import *

tau_exc = 10*ms
tau_pos = 20*ms
tau_neg = 20*ms
a_pos = 0.01
a_neg = -a_pos * tau_pos / tau_neg * 1.05
w_max = 1.0

def StdpSynapses(N, dt = None):
    update_model = '''
    w : 1
    dge/dt = -ge / tau_exc : 1
    ge_total_post = ge : 1 (summed)
    dstdp_pos/dt = -stdp_pos / tau_pos : 1
    dstdp_neg/dt = -stdp_neg / tau_neg : 1
    '''
    pre_model = '''
    stdp_pos += a_pos
    w = clip(w + stdp_neg, 0, w_max)
    ge = w
    '''
    post_model = '''
    stdp_neg += a_neg
    w = clip(w + stdp_pos, 0, w_max)
    ge = 0
    '''
    
    S = Synapses(N, model = update_model, pre = pre_model, post = post_model,\
        delay = {'post': 1*ms}, dt = dt)
    return S

if __name__ == "__main__":
    import scipy
    from lif import *
    from syn import *
    
    prefs.codegen.target = 'weave'
    defaultclock.dt = 1*ms
    
    N = LifNeurons(2)
    IN = NoiseInput(N, 0.025)
    SE = StdpSynapses(N)
    SE.connect(0, 1)
    state_monitor1 = StateMonitor(N, 'v', record = True)
    state_monitor2 = StateMonitor(SE, ('stdp_pos', 'stdp_neg'), record = True)
    network = Network()
    network.add(N, IN, SE, state_monitor1, state_monitor2)
    network.run(10*second, report = 'stdout', report_period = 10*second)
    
    figure(figsize = (4, 4))
    subplot(211)
    title("Membrane Potentials of Two Connected Neurons")
    plot(state_monitor1.t/ms, state_monitor1.v[0]/mV)
    plot(state_monitor1.t/ms, state_monitor1.v[1]/mV)
    subplot(212)
    title("Positive and Negative STDP Traces for Synapse Between Two Neurons")
    plot(state_monitor2.t/ms, state_monitor2.stdp_pos[0])
    plot(state_monitor2.t/ms, state_monitor2.stdp_neg[0])
    show()
    
    duration = 100*second
    n = 1000
    ne = 800
    
    N = LifNeurons(n)
    IN = NoiseInput(N, 0.025)
    SE = StdpSynapses(N)
    SE.connect('i < ne and i != j', p = 0.1)
    SE.w = 'rand() * 0.75'
    SI = InhibitorySynapses(N)
    SI.connect('i >= ne and i != j', p = 0.1)
    SI.w = -1.0
    
    rate_monitor = PopulationRateMonitor(N)
    
    figure(figsize=(12,4))
    subplot(131)
    title("Synaptic Weight Distribution Before Simulation")
    hist(SE.w / w_max, 20)
    xlabel('Weight / Maximum')
    ylabel('Count')
    
    network = Network()
    network.add(N, IN, SE, SI, rate_monitor)
    network.run(duration, report = 'stdout', report_period = 1*second)
    
    subplot(132)
    title("Firing Rate During Simulation")
    rates = scipy.stats.binned_statistic(rate_monitor.t/second, rate_monitor.rate, bins = int(duration / 1.0*second))
    plot(rates[1][1:], rates[0])
    xlabel('Time (s)')
    ylabel('Firing Rate (Hz)')
    
    subplot(133)
    title("Synaptic Weight Distribution After Simulation")
    hist(SE.w / w_max, 20)
    xlabel('Weight / Maximum')
    ylabel('Count')
    
    show()
