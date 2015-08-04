#!/usr/bin/env python
# -*- coding: utf-8 -*-
from lif import *
from syn import *
from da_stdp import *
import learnability

def get_param_search_levels(name, slice_levels=None):
    from itertools import product
    duration = 3600*second
    input_levels = [0.0275, 0.03, 0.0325, 0.0375, 0.05]
    connectivity_levels = [0.05, 0.1, 0.2]
    tau_rval_levels = [100*ms]
    rval_thresh_levels = [2.0]
    reward_amount_levels = [1*ms, 10*ms, 100*ms]
    reward_duration_levels = [100*ms]
    bias_levels = [-0.05]
    levels = list(product(input_levels, connectivity_levels, tau_rval_levels, rval_thresh_levels,
                          reward_amount_levels, reward_duration_levels, bias_levels))
    inputs = map(lambda i,x: (name, i, duration, x[0], x[1], x[2], x[3], x[4], x[5], x[6]),
                  range(0, len(levels)), levels)
    if slice_levels is None:
        slice_levels = (0, len(inputs))
    return inputs[slice_levels[0]:slice_levels[1]]

def run_fetzbaker(name, index, duration, input_rate, connectivity, tau_rval, rval_thresh, reward_amount, reward_duration, stdp_bias):
    params = LifParams(noise_scale=input_rate)
    params.update(SynParams())
    params.update(DaStdpParams(a_neg=-1.0 + stdp_bias, connectivity=connectivity))
    params.update({'tau_rval': tau_rval, 'rval_thresh': rval_thresh})
    neurons, excitatory_synapses, inhibitory_synapses, network = learnability.setup_network(params)
    reward_threshold = NeuronGroup(1, model="drval/dt = -rval / tau_rval : 1 (unless refractory)",
                               threshold="rval >= rval_thresh", reset="rval = 0", refractory=reward_duration,
                               namespace=params)
    reward_increment = Synapses(neurons, reward_threshold, model='inc : 1',
                                pre='rval += inc', post='inc *= 0.99', connect='i==0')
    reward_increment.inc = 1.0
    reward_function = lambda synapses: 0 if reward_threshold.not_refractory[0] else (reward_amount / reward_duration)
    reward_link = RewardUnit(excitatory_synapses, reward_function)
    network.add(reward_threshold, reward_increment, reward_link)
    learnability.run_sim("fetzbaker/" + name, index, duration, neurons, excitatory_synapses, network, params)
