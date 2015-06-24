#!/usr/bin/env python
# -*- coding: utf-8 -*-
from lif import *
from syn import *
from da_stdp import *

timestep = 1.0*ms
neuron_count = 1000
excitatory_count = 800
excitatory_weight = 0.25
inhibitory_weight = -1.0
tau_activity = 1*second
activity_increment = 0.5
activity_decrement = activity_increment / excitatory_count
activity_threshold = 1.0
reward_duration = 100*ms
tau_reward = 10*ms

def run_sim(name, index, duration, input_rate, connectivity, reward_amount):
    import random
    prefs.codegen.target = 'weave'
    defaultclock.dt = timestep
    print("Fetz-Baker Simulation {0} Started: duration={1}, input={2}, connectivity={3}, reward={4}".format(
          index, duration, input_rate, connectivity, reward_amount))    
    N = LifNeurons(neuron_count)
    IN = NoiseInput(N, input_rate)
    SE = DaStdpSynapses(N)
    synapse_count = int(neuron_count * connectivity)
    for i in range(0, excitatory_count):
        SE.connect(i, random.sample(range(0, neuron_count), synapse_count))
    #SE.connect('i < excitatory_count and i != j', p=connectivity)
    SE.w = 'excitatory_weight'
    SI = InhibitorySynapses(N)
    for i in range(excitatory_count, neuron_count):
        SI.connect(i, random.sample(range(0, neuron_count), synapse_count))
    #SI.connect('i >= excitatory_count and i != j', p=connectivity)
    SI.w = inhibitory_weight
    activity_integrator_model = '''
    dactivity/dt = -activity / tau_activity : 1 (unless refractory)
    dr/dt = -r / tau_reward : 1
    '''
    reset_model = '''
    activity = 0
    r += reward_amount
    '''
    AI = NeuronGroup(1, model=activity_integrator_model, threshold='activity > activity_threshold', reset=reset_model, refractory=reward_duration)
    AI.activity = 0
    SEAI = Synapses(N, AI, pre='activity += activity_increment', connect='i == 0')
    SIAI = Synapses(N, AI, pre='activity -= activity_decrement', connect='i != 0 and i < excitatory_count')
    @network_operation(dt=timestep)
    def R():
        SE.r = AI.r[0]
    rate_monitor = PopulationRateMonitor(N)
    state_monitor = StateMonitor(AI, ('activity', 'not_refractory'), record=[0])
    state_monitor2 = StateMonitor(SE, 'r', record=[0])
    spike_monitor = SpikeMonitor(N)
    network = Network()
    network.add(N, IN, SE, SI, AI, SEAI, SIAI, R, rate_monitor, state_monitor, state_monitor2, spike_monitor)
    network.run(duration, report='stdout', report_period=60*second)
    w_post = SE.w
    datafile = open("fetzbaker/{0}_{1}.dat".format(name, index), 'wb')
    numpy.savez(datafile, duration=duration, input_rate=input_rate, connectivity=connectivity, reward_amount=reward_amount,
                w_post=w_post, t=state_monitor.t/second, activity=state_monitor.activity[0], r=state_monitor2.r[0],
                spikes_t=spike_monitor.t, spikes_i=spike_monitor.i, rate=rate_monitor.rate)
    datafile.close()
    print("Fetz-Baker Simulation {0} Ended".format(index))

def parallel_function(fn):
    def easy_parallelize(fn, inputs):
        from multiprocessing import Pool
        pool = Pool(processes=4)
        pool.map(fn, inputs)
        pool.close()
        pool.join()
    from functools import partial
    return partial(easy_parallelize, fn)

def packed_run_sim(packed_args):
    run_sim(*packed_args)
run_sim.parallel = parallel_function(packed_run_sim)

def plot_sim(name, index):
    from scipy import stats
    datafile = open("fetzbaker/{0}_{1}.dat".format(name, index), 'rb')
    data = numpy.load(datafile)
    figure(figsize=(12,12))
    suptitle("Duration: {0}, Input: {1}, Connectivity: {2}, Reward: {3}".format(
             data['duration'], data['input_rate'], data['connectivity'], data['reward_amount']))
    subplot(221)
    plot(data['spikes_t'], data['spikes_i'], '.k')
    xlabel("Time (s)")
    ylabel("Neuron Id")
    subplot(222)
    plot(data['t'], data['activity'], label='activity')
    plot(data['t'], data['r'], label='reward')
    legend()
    xlabel("Time (s)")
    ylabel("Activity Integrator")
    subplot(223)
    rates = stats.binned_statistic(data['t'], data['rate'], bins=int(data['duration'] / 1.0*second))
    plot(rates[1][1:], rates[0])
    xlabel("Time (s)")
    ylabel("Firing Rate (Hz)")
    subplot(224)
    hist(data['w_post'] / w_max, 20)
    xlabel("Weight / Maximum")
    ylabel("Count")
    show()
    datafile.close()

def param_search(name):
    from itertools import product, imap, izip
    duration = 1800*second
    input_levels = [0.025, 0.0275, 0.03, 0.0325]
    connectivity_levels = [0.1]
    reward_levels = [1.0]
    num_simulations = len(input_levels) * len(connectivity_levels) * len(reward_levels)
    levels = product(input_levels, connectivity_levels, reward_levels)
    inputs = imap(lambda i,x: (name, i, duration, x[0], x[1], x[2]),
                  range(0, num_simulations), levels)
    run_sim.parallel(inputs)
    #for e in inputs:
        #print("Running Simulation {0}: {1}".format(simulation_number, e))
