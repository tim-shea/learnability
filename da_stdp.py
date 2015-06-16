# -*- coding: utf-8 -*-
from brian2 import *

tau_exc = 10*ms
tau_pos = 20*ms
tau_neg = 20*ms
tau_elig = 1000*ms
a_pos = 1.0
a_neg = -a_pos * tau_pos / tau_neg * 1.05
w_max = 1.0

def DaStdpSynapses(N, dt = None):
    update_model = """
    r : 1 (shared)
    dl/dt = -l/tau_elig : 1
    dw/dt = l * r : 1
    dge/dt = -ge / tau_exc : 1
    ge_total_post = ge : 1 (summed)
    dstdp_pos/dt = -stdp_pos / tau_pos : 1 (event-driven)
    dstdp_neg/dt = -stdp_neg / tau_neg : 1 (event-driven)
    """
    pre_model = """
    stdp_pos += a_pos
    l += stdp_neg
    w = clip(w, 0, w_max)
    ge = w
    """
    post_model = """
    stdp_neg += a_neg
    l += stdp_pos
    ge = 0
    """
    
    S = Synapses(N, model = update_model, pre = pre_model, post = post_model,\
        delay = {'post': 1*ms}, dt = dt)
    return S

def RewardUnit(S, reward_function, dt = None):
    @network_operation(dt = dt)
    def R():
        S.r = reward_function(S)
    return R

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
    
    N = LifNeurons(n)
    IN = NoiseInput(N, 0.025)
    SE = DaStdpSynapses(N)
    SE.connect('i < ne and i != j', p = 0.1)
    SE.w = 'rand() * 0.75'
    SI = InhibitorySynapses(N)
    SI.connect('i >= ne and i != j', p = 0.1)
    SI.w = -1.0
    def reward_function(S):
        return 0.002 + (S.r - 0.002) * 0.9 + (1 if int(S.t/ms) % 2000 == 1000 else 0)
    R = RewardUnit(SE, reward_function, 10*ms)
    
    state_monitor = StateMonitor(SE, ('w', 'l', 'r'), [0, 1, 2, 3, 4])
    
    network = Network()
    network.add(N, IN, SE, SI, R, state_monitor)
    
    figure()
    
    subplot(131)
    title("Synaptic Weight Distribution Before Simulation")
    hist(SE.w / w_max, 20)
    xlabel('Weight / Maximum')
    ylabel('Count')
    
    network.run(duration, report = 'stdout', report_period = binsize)
    
    subplot(132)
    title("Synapse Traces During Simulation")
    plot(state_monitor.t/second, state_monitor.w[0], label="Synapse 0")
    plot(state_monitor.t/second, state_monitor.w[1], label="Synapse 1")
    plot(state_monitor.t/second, state_monitor.w[2], label="Synapse 2")
    plot(state_monitor.t/second, state_monitor.w[3], label="Synapse 3")
    plot(state_monitor.t/second, state_monitor.w[4], label="Synapse 4")
    plot(state_monitor.t/second, state_monitor.r[0], label="Dopamine")
    xlabel('Time (s)')
    ylabel('Weight')
    
    subplot(133)
    title("Synaptic Weight Distribution After Simulation")
    hist(SE.w / w_max, 20)
    xlabel('Weight / Maximum')
    ylabel('Count')
    
    show()
