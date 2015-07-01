#!/usr/bin/env python
# -*- coding: utf-8 -*-
from lif import *
from syn import *
from da_stdp import *

def run_sim(name, index, duration, input_rate, connectivity, reward_rate, reward_amount, reward_duration, stdp_bias):
    prefs.codegen.target = 'weave'
    defaultclock.dt = 1.0*ms
    print("Stability Simulation {0} Started: dur={1}, in={2}, conn={3}, rwr={4}, rwa={5}, rwd={6}, bias={7}".format(
          index, duration, input_rate, connectivity, reward_rate, reward_amount, reward_duration, stdp_bias))
    params = LifParams(noise_scale=input_rate)
    params.update(SynParams())
    params.update(DaStdpParams(a_neg=-1.0 + stdp_bias))
    N = LifNeurons(1000, params)
    SE = DaStdpSynapses(N, params)
    SE.connect('i < 800 and i != j', p=connectivity)
    SE.w = 0.5
    SI = InhibitorySynapses(N)
    SI.connect('i >= 800 and i != j', p=connectivity)
    SI.w = -1.0
    reward_model = """
    drtimer/dt = reward_rate : 1 (unless refractory)
    """
    DA = NeuronGroup(1, model=reward_model, threshold="rtimer >= 1.0", reset="rtimer = 0", refractory=reward_duration)
    R = RewardUnit(SE, lambda S: 0 if DA.not_refractory[0] else (reward_amount / reward_duration))
    rate_monitor = PopulationRateMonitor(N)
    state_monitor = StateMonitor(SE, ('w', 'r'), record=numpy.random.choice(len(SE.i), size=5, replace=False))
    network = Network()
    network.add(N, SE, SI, DA, R, rate_monitor, state_monitor)
    periods = int(duration / 60*second)
    spikes_t = []
    spikes_i = []
    from timeit import default_timer
    real_start_time = default_timer()
    weights = ndarray((periods + 1, size(SE.w)))
    weights[0] = SE.w
    for period in range(1, periods + 1):
        spike_monitor = SpikeMonitor(N)
        network.add(spike_monitor)
        network.run(1*second)
        spikes_t.append(array(spike_monitor.t))
        spikes_i.append(array(spike_monitor.i))
        network.remove(spike_monitor)
        network.run(59*second)
        real_elapsed_time = default_timer() - real_start_time
        print("Stability Simulation {0}: {1} min. simulated ({2:%}) in {3} sec.".format(index, period, float(period) / periods, real_elapsed_time))
        weights[period] = SE.w
    datafile = open("stability/{0}_{1}.dat".format(name, index), 'wb')
    numpy.savez(datafile, duration=duration, input_rate=input_rate, connectivity=connectivity, reward_rate=reward_rate,
                reward_amount=reward_amount, reward_duration=reward_duration, stdp_bias=stdp_bias,
                t=rate_monitor.t, rate=rate_monitor.rate,
                w0=state_monitor.w[0], w1=state_monitor.w[1], w2=state_monitor.w[2], w3=state_monitor.w[3], w4=state_monitor.w[4],
                r=state_monitor.r[0], weights=weights, spikes_t=spikes_t, spikes_i=spikes_i)
    datafile.close()
    print("Stability Simulation {0} Ended".format(index))

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
    file = open("stability/{0}_{1}.dat".format(name, index), 'rb')
    data = numpy.load(file)
    figure(figsize=(12,12))
    suptitle("Duration: {0}, Input: {1}, Connectivity: {2}, Reward: {3}".format(
          data['duration'], data['input_rate'], data['connectivity'], data['reward_amount']))
    subplot(311)
    plot(data['t'], data['w0'])
    plot(data['t'], data['w1'])
    plot(data['t'], data['w2'])
    plot(data['t'], data['w3'])
    plot(data['t'], data['w4'])
    plot(data['t'], data['r'])
    xlabel("Time (s)")
    ylabel("Reward")
    subplot(312)
    rates = stats.binned_statistic(data['t'], data['rate'], bins=int(data['duration'] / 1.0*second))
    plot(rates[1][1:], rates[0])
    xlabel("Time (s)")
    ylabel("Firing Rate (Hz)")
    subplot(313)
    means = mean(data['weights'], axis=1)
    upper = percentile(data['weights'], 75, axis=1)
    lower = percentile(data['weights'], 25, axis=1)
    num_periods = int(data['duration'] / 60)
    plot(range(num_periods), means, '-r')
    plot(range(num_periods), upper, '-k')
    plot(range(num_periods), lower, '-k')
    xlabel("Period (min.)")
    ylabel("Count")
    show()
    file.close()

def param_search(name, slice_inputs=None):
    from itertools import product, imap, izip
    duration = 300*second
    input_levels = [0.02, 0.025, 0.0375, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
    connectivity_levels = [0.01, 0.025, 0.05, 0.1, 0.125, 0.25]
    reward_rate_levels = [1.0/(900*ms)]
    reward_amount_levels = [10*ms, 100*ms]
    reward_duration_levels = [100*ms]
    bias_levels = [-0.05]
    levels = list(product(input_levels, connectivity_levels, reward_rate_levels, reward_amount_levels, reward_duration_levels, bias_levels))
    inputs = map(lambda i,x: (name, i, duration, x[0], x[1], x[2], x[3], x[4], x[5]),
                  range(0, len(levels)), levels)
    if slice_inputs is None:
        slice_inputs = (0, len(inputs))
    run_sim.parallel(inputs[slice_inputs[0]:slice_inputs[1]])

def plot_param_search(name):
    import os.path
    import scipy.stats
    fileset = []
    dataset = []
    index = 0
    while os.path.exists("stability/{0}_{1}.dat".format(name, index)):
        file = open("stability/{0}_{1}.dat".format(name, index), 'rb')
        fileset.append(file)
        data = numpy.load(file)
        dataset.append(data)
        index += 1
    figure()
    subplot(221)
    for data in dataset:
        plot(data['t'], data['w0'] / DaStdpParams()['w_max'], label="IN: {0} CONN: {1}".format(data['input_rate'], data['connectivity']))
    xlabel("Time (s)")
    ylabel("Target Synapse Weight")
    subplot(222)
    for data in dataset:
        plot(data['t'], cumsum(data['rew']), label="IN: {0} CONN: {1}".format(data['input_rate'], data['connectivity']))
    xlabel("Time (s)")
    ylabel("Cumulative Reward")
    subplot(223)
    for data in dataset:
        rates = scipy.stats.binned_statistic(data['t'], data['rate'], bins=int(data['duration'] / 10.0*second))
        plot(rates[1][1:], rates[0], label="IN: {0} CONN: {1}".format(data['input_rate'], data['connectivity']))
    xlabel("Time (s)")
    ylabel("Firing Rate (Hz)")
    subplot(224)
    for data in dataset:
	    hist(data['w_post'] / DaStdpParams()['w_max'], 20, histtype='step')
    xlabel("Weight / Maximum")
    ylabel("Count")
    for file in fileset:
        file.close()
    show()
