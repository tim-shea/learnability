# -*- coding: utf-8 -*-
from lif import *
from syn import *
from da_stdp import *

trials = 100
duration = 10*second * trials
timestep = 1.0*ms
neuron_count = 1000
excitatory_count = 800
input_rate = 0.025
minimum_weight = 0.0
maximum_weight = 0.25
inhibitory_weight = -0.25
connectivity = 0.1

stimulus_size = 50
response_size = 50
reward_amount = 0.5
reward_decay = 0.99
coincidence_window = 20*ms
reward_delay = 1*second

prefs.codegen.target = 'numpy'
defaultclock.dt = timestep

N = LifNeurons(neuron_count)

input_model = """
dr/dt = (-r + rand())/dt : 1
noise = r * (%f + int(t/ms %% 10000 < 10) * int(i < stimulus_size) * 0.05) : 1
I = (eq_exc - v_reset) * tau_mem/ms * noise : 1
""" % input_rate
IN = NeuronGroup(neuron_count, model = input_model)
N.I = linked_var(IN, 'I')

response_model = '''
reward : second (shared)
response : second (shared)
ds/dt = 0/ms : 1
'''
R = NeuronGroup(2, model = response_model, method = 'euler')
R.reward = -1*ms
R.response = 20*ms
SR = Synapses(N, R, pre = 's += 1', connect = 'i >= stimulus_size and j == floor((i - stimulus_size) / response_size)')

responses = []
@network_operation(dt = 1*ms)
def response_model():
	if int(R.t/ms) % 10000 == 0:
		R.s = 1
	if int(R.t/ms) % 10000 == 20:
		responses.append(0 if R.s[0] == R.s[1] else 1 if R.s[0] > R.s[1] else 2)
		if responses[-1] == 1:
			R.reward = R.t + reward_delay / (R.s[0] / R.s[1])

SE = DaStdpSynapses(N)
SE.connect('i < excitatory_count and i != j', p = connectivity)
SE.w = 'minimum_weight + rand() * (maximum_weight - minimum_weight)'
SI = InhibitorySynapses(N)
SI.connect('i >= excitatory_count and i != j', p = connectivity)
SI.w = inhibitory_weight

def reward_function(S):
	return SE.r * reward_decay + (reward_amount if floor(R.t/ms) == floor(R.reward/ms) else 0)
REW = RewardUnit(SE, reward_function)

state_monitor = StateMonitor(SE, 'r', record = [0])

network = Network()
network.add(N, IN, R, SR, SE, SI, REW, response_model, state_monitor)
network.run(duration, report = 'stdout', report_period = 10*second)

figure(figsize=(8,4))
subplot(211)
plot(responses, '.k')
subplot(212)
plot(state_monitor.t/ms, state_monitor.r[0])
show()
