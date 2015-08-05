#!/usr/bin/env python
# -*- coding: utf-8 -*-
from lif import *
from syn import *
from da_stdp import *
import learnability

def get_param_search_levels(name, slice_levels=None):
    from itertools import product
    duration = 3600*second
    input_levels = [0.025, 0.03, 0.0375, 0.05]
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

def setup(input_rate, connectivity, reward_delay, reward_amount, reward_duration, stdp_bias):
    params = LifParams(noise_scale=input_rate)
    params.update(SynParams())
    params.update(DaStdpParams(a_neg=-1.0 + stdp_bias, connectivity=connectivity))
    params.update({'reward_delay': reward_delay})
    neurons, excitatory_synapses, inhibitory_synapses, network = learnability.setup_network(params)
    
    stimulus = NeuronGroup(1, model="dtimer/dt = 1/second : 1", threshold="timer > 1", reset="timer = 0", namespace=params)
    stimulus_synapse_model = """
    tstimulus : second
    ge_total_post = 2.5 * int(t - tstimulus < 20*ms) : 1 (summed)
    """
    stimulus_synapses = Synapses(stimulus, neurons, model=stimulus_synapse_model, pre="tstimulus = t", connect="j < 50")
    stimulus_synapses.tstimulus = -1*second
    
    response_model = """
    tresponse : second
    response_one : 1
    response_two : 1
    response_rate = response_one / (response_two + 1) : 1
    dreward/dt = response_rate / reward_delay : 1 (unless refractory)
    """
    response_reset = """
    response_one = 0
    response_two = 0
    reward = 0
    """
    response = NeuronGroup(1, model=response_model, threshold="reward > 1", reset=response_reset,
                           refractory=reward_duration, namespace=params)
    response.tresponse = -1*second
    response.response_one = 0
    response.response_two = 0
    
    s_r_pre = """
    tresponse = t
    response_one = 0
    response_two = 0
    """
    s_r_synapse = Synapses(stimulus, response, pre=s_r_pre, connect=True)
    response_synapse_pre = """
    response_one += 1 * int(i > 50 and i < 100) * int(t - tresponse < 20*ms)
    response_two += 1 * int(i > 100) * int(t - tresponse < 20*ms)
    """
    response_synapses = Synapses(neurons, response, pre=response_synapse_pre, connect="i > 50 and i < 150")
    reward_function = lambda synapses: 0 if response.not_refractory[0] else (reward_amount / reward_duration)
    reward_link = RewardUnit(excitatory_synapses, reward_function)
    network.add(stimulus, stimulus_synapses, response, s_r_synapse, response_synapses, reward_link)
    return neurons, excitatory_synapses, network, params

def run(name, index, duration, neurons, excitatory_synapses, network, params):
    learnability.run_sim("izhikevich3/" + name, index, duration, neurons, excitatory_synapses, network, params)

def setup_and_run(name, index, duration, input_rate, connectivity, reward_delay, reward_amount, reward_duration, stdp_bias):
    neurons, excitatory_synapses, network, params = setup(input_rate, connectivity, reward_delay, reward_amount, reward_duration, stdp_bias)
    run(name, index, duration, neurons, excitatory_synapses, network, params)

if __name__ == "__main__":
    run_parallel(setup_and_run, get_param_search_levels('initial', (0,8)))
