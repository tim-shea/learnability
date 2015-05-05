# -*- coding: utf-8 -*-
from da_stdp import *

duration = 3600*second
timestep = 1.0*ms
neuron_count = 1000
excitatory_count = 800
input_rate = 0.001
minimum_excitatory_weight = 0.0
maximum_excitatory_weight = 0.25
inhibitory_weight = -0.25
connectivity = 0.1
reward_amount = 0.5
reward_decay = 0.99
coincidence_window = 20*ms
reward_delay = 0.5*second

def run_sim(id):
prefs.codegen.target = 'weave'
defaultclock.dt = timestep

N = LifNeurons(neuron_count)
IN = BernoulliSpikeInput(N, input_rate)
SE = DaStdpSynapses(N)
SE.connect('i < excitatory_count and i != j', p = connectivity)
SE.w = 'minimum_excitatory_weight + rand() * (maximum_excitatory_weight - minimum_excitatory_weight)'
SI = Synapses(N, pre = 'gi_post += inhibitory_weight')
SI.connect('i >= excitatory_count and i != j', p = connectivity)

reward_model = '''
reward = (t + reward_delay) * int(s > firing_threshold) : 1 (shared)
ds/dt = 0 : 1
'''
NR = NeuronGroup(1, reward_model)
SR = Synapses(N, NR, pre = 's += 1', connect = 'i == 0')

def reward_function(S):
	return S.r * reward_decay + (reward_amount if int(S.t/ms) == int((D.detected + reward_delay)/ms) else 0)
R = RewardUnit(SE, reward_function)

rate_monitor = PopulationRateMonitor(N[0:1])
state_monitor = StateMonitor(N, 'r', record = [0])

network = Network()
network.add(N, IN, SE, SI, D, R, rate_monitor, state_monitor)
network.run(duration, report = 'stdout', report_period = 10*second)
