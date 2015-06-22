#!/usr/bin/env python
# -*- coding: utf-8 -*-
from lif import *
from syn import *
from da_stdp import *

timestep = 1.0*ms
neuron_count = 1000
excitatory_count = 800
minimum_excitatory_weight = 0.0
maximum_excitatory_weight = 0.75
inhibitory_weight = -1.0
reward_decay = 0.99
coincidence_window = 20*ms
reward_delay = 0.5*second

def run_sim(id, duration, input_rate, connectivity, reward_amount):
    prefs.codegen.target = 'numpy'
    defaultclock.dt = timestep
    
    print("Running Simulation {0}: duration={1}, input={2}, connectivity={3}, reward={4}".format(
          id, duration, input_rate, connectivity, reward_amount))
    
    N = LifNeurons(neuron_count)
    IN = NoiseInput(N, input_rate)
    SE = DaStdpSynapses(N)
    SE.connect('i < excitatory_count and i != j', p=connectivity)
    SE.w = 'minimum_excitatory_weight + rand() * (maximum_excitatory_weight - minimum_excitatory_weight)'
    SI = InhibitorySynapses(N)
    SI.connect('i >= excitatory_count and i != j', p=connectivity)
    SI.w = inhibitory_weight
    
    det_model = '''
    pre_spike : second
    detected : second
    '''
    det_pre = 'pre_spike = t'
    det_post = '''
    set_detected = int(t - pre_spike < coincidence_window)
    detected = detected * (1 - set_detected) + t * set_detected
    '''
    
    SE.w[0] = 0
    source = SE.i[0]
    target = SE.j[0]
    
    D = Synapses(N, model=det_model, pre=det_pre, post=det_post, connect='i == source and j == target')
    D.detected = -reward_delay
    
    def reward_function(S):
        return S.r * reward_decay + (reward_amount if int(S.t/ms) == int((D.detected + reward_delay)/ms) else 0)
    R = RewardUnit(SE, reward_function)
    
    rate_monitor = PopulationRateMonitor(N)
    state_monitor = StateMonitor(SE, ('r', 'l', 'w'), record=[0])
    
    network = Network()
    network.add(N, IN, SE, SI, D, R, rate_monitor, state_monitor)
    network.run(duration, report='stdout', report_period=10*second)
    w_post = SE.w
    
    datafile = open("%s.dat" % id, 'wb')
    numpy.savez(datafile, w_post=w_post, t=state_monitor.t/second, w0=state_monitor.w[0],
                l0=state_monitor.l[0], rew=state_monitor.r[0], rate=rate_monitor.rate)
    datafile.close()
    return (id, True)

def parallel_function(fn):
    def easy_parallelize(fn, inputs):
        """ assumes f takes sequence as input, easy w/ Python's scope """
        from multiprocessing import Pool
        pool = Pool(processes=4)
        result = pool.map(fn, inputs)
        for r in result:
            if r is not None and r[1]:
                print("Simulation Cleaned {0}".format(r[0]))
            else:
                print("Simulation Failed {0}".format(r[0]))
        pool.close()
        pool.join()
    from functools import partial
    return partial(easy_parallelize, fn)

def packed_run_sim(packed_args):
    run_sim(*packed_args)
run_sim.parallel = parallel_function(packed_run_sim)

def plot_sim(id, duration):
    from scipy import stats
    
    datafile = open("%s.dat" % id, 'rb')
    data = numpy.load(datafile)
    
    figure(figsize=(12,12))
    subplot(221)
    plot(data['t'], data['w0'] / w_max)
    plot(data['t'], data['l0'])
    xlabel('Time (s)')
    ylabel('Weight/Eligibility')
    
    subplot(222)
    plot(data['t'], cumsum(data['rew']))
    xlabel('Time (s)')
    ylabel('Cumulative Reward')
    
    subplot(223)
    rates = stats.binned_statistic(data['t'], data['rate'], bins=int(duration / 1.0*second))
    plot(rates[1][1:], rates[0])
    xlabel('Time (s)')
    ylabel('Firing Rate (Hz)')
    
    subplot(224)
    hist(data['w_post'] / w_max, 20)
    xlabel('Weight / Maximum')
    ylabel('Count')
    show()
    
    datafile.close()
