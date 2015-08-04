#!/usr/bin/env python
# -*- coding: utf-8 -*-
from lif import *
from syn import *
from da_stdp import *
import learnability

def get_param_search_levels(name, slice_levels=None):
    from itertools import product
    duration = 3600*second
    input_levels = [0.025, 0.0375, 0.05, 0.075, 0.1]
    connectivity_levels = [0.05, 0.1, 0.2]
    reward_delay_levels = [500*ms]
    reward_amount_levels = [10*ms, 100*ms]
    reward_duration_levels = [100*ms]
    bias_levels = [-0.05]
    levels = list(product(input_levels, connectivity_levels, reward_delay_levels,
                          reward_amount_levels, reward_duration_levels, bias_levels))
    inputs = map(lambda i,x: (name, i, duration, x[0], x[1], x[2], x[3], x[4], [5]),
                  range(0, len(levels)), levels)
    if slice_levels is None:
        slice_levels = (0, len(inputs))
    return inputs[slice_levels[0]:slice_levels[1]]

def run(name, index, duration, input_rate, connectivity, reward_delay, reward_amount, reward_duration, stdp_bias):
    params = LifParams(noise_scale=input_rate)
    params.update(SynParams())
    params.update(DaStdpParams(a_neg=-1.0 + stdp_bias, connectivity=connectivity))
    params.update({'reward_delay': reward_delay})
    neurons, excitatory_synapses, inhibitory_synapses, network = learnability.setup_network(params)
    stimulus_model = """
    
    response_model = """    
    response_period : 1
    response_one : 1
    response_two : 1
    response_rate = response_one / (response_two + 1) : 1 (linked)
    dstim
    """
    response_model = """
    drewtimer/dt = response_rate / reward_delay : 1 (unless refractory)
    """
    response_unit = NeuronGroup(1, model=response_model, threshold="rewtimer >= 1",
                               reset="rtimer = 0", refractory=reward_duration,
                               namespace=params)
    response_link_model = "response_period : 1 (shared)"
    response_link_pre = """
    response_one += 1 * int(i < 50) * response_period
    response_two += 1 * int(i > 50) * response_period
    """
    response_link = Synapses(neurons, reward_timer, model=response_link_model, pre=reward_link_pre, connect="i < 100")
    response_link.response_period = 0
    reward_function = lambda synapses: 0 if reward_threshold.not_refractory[0] else (reward_amount / reward_duration)
    reward_link = RewardUnit(excitatory_synapses, reward_function)
    network.add(reward_timer, response_link, reward_link)
    learnability.run_sim("izhikevich3/" + name, index, duration, neurons, excitatory_synapses, network, params)
