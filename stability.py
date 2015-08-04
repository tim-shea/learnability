#!/usr/bin/env python
# -*- coding: utf-8 -*-
from lif import *
from syn import *
from da_stdp import *
import learnability

def get_param_search_levels(name, slice_levels=None):
    from itertools import product
    duration = 300*second
    input_levels = [0.02, 0.025, 0.0375, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
    connectivity_levels = [0.01, 0.025, 0.05, 0.1, 0.125, 0.25]
    reward_rate_levels = [1.0/(900*ms)]
    reward_amount_levels = [10*ms, 100*ms]
    reward_duration_levels = [100*ms]
    bias_levels = [-0.05]
    levels = list(product(input_levels, connectivity_levels, reward_rate_levels,
                          reward_amount_levels, reward_duration_levels, bias_levels))
    inputs = map(lambda i,x: (name, i, duration, x[0], x[1], x[2], x[3], x[4], x[5]),
                  range(0, len(levels)), levels)
    if slice_levels is None:
        slice_levels = (0, len(inputs))
    return inputs[slice_levels[0]:slice_levels[1]]

def run(name, index, duration, input_rate, connectivity, reward_rate, reward_amount, reward_duration, stdp_bias):
    params = LifParams(noise_scale=input_rate)
    params.update(SynParams())
    params.update(DaStdpParams(a_neg=-1.0 + stdp_bias, connectivity=connectivity))
    params.update({'reward_rate': reward_rate})
    neurons, excitatory_synapses, inhibitory_synapses, network = learnability.setup_network(params)
    reward_model = "drtimer/dt = reward_rate : 1 (unless refractory)"
    reward_timer = NeuronGroup(1, model=reward_model, threshold="rtimer >= 1.0",
                               reset="rtimer = 0", refractory=reward_duration,
                               namespace=params)
    reward_function = lambda synapses: 0 if reward_timer.not_refractory[0] else (reward_amount / reward_duration)
    reward_unit = RewardUnit(excitatory_synapses, reward_function)
    network.add(reward_timer, reward_unit)
    learnability.run_sim("stability/" + name, index, duration, neurons, excitatory_synapses, network, params)

def plot_stability(name):
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
    subplot(211)
    values = array([(data['input_rate'], data['connectivity'], mean(data['rate']), std(data['rate'])) for data in dataset])
    colors = (values[:,1] - min(values[:,1])) / max(values[:,1])
    scatter(values[:,2], values[:,3], c=colors, alpha=0.25, edgecolors='none', cmap=get_cmap('rainbow'), s=50)
    legend(unique(values[:,1]), )
    xlabel("Firing Rate (Hz)")
    ylabel("Standard Deviation")
    subplot(212)
    scatter(values[:,0], values[:,3] / values[:,2], c=colors, cmap=get_cmap('hsv'), s=50)
    xlabel("Input Rate")
    ylabel("Coefficient of Variation")
    for file in fileset:
        file.close()
    show()
