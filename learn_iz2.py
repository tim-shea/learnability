# -*- coding: utf-8 -*-
import random as rnd
from scipy import signal
from lif import *
from syn import *
from da_stdp import *

timestep = 1.0*ms
neuron_count = 500
excitatory_count = 400
input_rate = 0.025
minimum_weight = 0.0
maximum_weight = 0.5
inhibitory_weight = -0.5
connectivity = 0.1

tau_pre = 20*ms
tau_post = 20*ms
tau_elig = 100*ms

stimulus_size = 25
stimulus_count = 20
minimum_isi = 100*ms
maximum_isi = 300*ms
reward_amount = 1.0
reward_decay = 0.99
reward_delay = 250*ms

prefs.codegen.target = 'weave'
defaultclock.dt = timestep

N = LifNeurons(neuron_count)
stimulus_array = zeros((stimulus_count, neuron_count))
for x in range(0,stimulus_count):
	for y in rnd.sample(range(0, excitatory_count), stimulus_size):
		stimulus_array[x,y] = 1
stimuli = TimedArray(stimulus_array, dt = 1*ms)

input_model = """
reward : second (shared)
stimulus : second (shared)
t_stimulus : second (shared)
dr/dt = (-r + rand())/dt : 1
inject_spike = int(t >= t_stimulus) * stimuli(stimulus, i) : 1
I = (eq_exc - v_reset) * tau_mem/ms * (inject_spike + r * %f) : 1
""" % input_rate
IN = NeuronGroup(neuron_count, model = input_model)
IN.stimulus = 0*ms
IN.t_stimulus = minimum_isi
N.I = linked_var(IN, 'I')

targets = empty(0)
rewards = empty(0)
@network_operation(dt = 1*ms, when = "resets")
def next_stimulus():
	global targets, rewards
	if (IN.t > IN.t_stimulus):
		if (IN.stimulus == 0*ms):
			IN.reward = IN.t + 10*ms + random() * reward_delay
			targets = append(targets, IN.t/second)
			rewards = append(rewards, IN.reward/second)
		IN.t_stimulus = N.t + minimum_isi + (maximum_isi - minimum_isi) * random()
		IN.stimulus = floor(random() * stimulus_count)*ms

SE = DaStdpSynapses(N)
SE.connect('i < excitatory_count and i != j', p = connectivity)
SE.w = 'minimum_weight + rand() * (maximum_weight - minimum_weight)'
SI = InhibitorySynapses(N)
SI.connect('i >= excitatory_count and i != j', p = connectivity)
SI.w = inhibitory_weight

def reward_function(S):
	return SE.r * reward_decay + (reward_amount if floor(S.t/ms) == floor(IN.reward/ms) else 0)
R = RewardUnit(SE, reward_function)

rate_monitor = PopulationRateMonitor(N)
spike_monitor = SpikeMonitor(N)
state_monitor = StateMonitor(SE, ('r', 'l', 'w'), record = True)

network = Network()
network.add(N, IN, SE, SI, R, next_stimulus, rate_monitor, spike_monitor, state_monitor)

pre_weights = zeros(stimulus_count)
for stimulus in range(0, stimulus_count):
	for synapse in range(0, asarray(SE.j).size):
		if stimulus_array[stimulus,SE.j[synapse]] == 1:
			pre_weights[stimulus] += SE.w[synapse]

network.run(10*second)
figure()
subplot(411)
plot(spike_monitor.t/second, spike_monitor.i, ',k')
clipped_targets = numpy.array(targets)
clipped_targets = clipped_targets[clipped_targets < 10]
plot(clipped_targets, ones_like(clipped_targets) * -10, 'ob')
clipped_rewards = numpy.array(rewards)
clipped_rewards = clipped_rewards[clipped_rewards < 10]
plot(clipped_rewards, ones_like(clipped_rewards) * -10, '^r')

network.remove(spike_monitor)
network.run(300*second, report = 'stdout', report_period = 10*second)

spike_monitor = SpikeMonitor(N)
network.add(spike_monitor)
network.run(10*second)
subplot(412)
plot(spike_monitor.t/second, spike_monitor.i, ',k')
clipped_targets = numpy.array(targets)
clipped_targets = clipped_targets[logical_and(clipped_targets >= 310, clipped_targets < 320)]
plot(clipped_targets, ones_like(clipped_targets) * -10, 'ob')
clipped_rewards = numpy.array(rewards)
clipped_rewards = clipped_rewards[logical_and(clipped_rewards >= 310, clipped_rewards < 320)]
plot(clipped_rewards, ones_like(clipped_rewards) * -10, '^r')

subplot(413)
plot(state_monitor.t/second, state_monitor.r[0]/second, label = 'reward')
plot(state_monitor.t/second, sum(state_monitor.l, axis = 0)/asarray(SE.l).size, label = 'mean eligibility')
plot(state_monitor.t/second, sum(state_monitor.w, axis = 0)/asarray(SE.w).size, label = 'mean weight')
legend()

post_weights = zeros(stimulus_count)
for stimulus in range(0, stimulus_count):
	for synapse in range(0, asarray(SE.j).size):
		if stimulus_array[stimulus,SE.j[synapse]] == 1:
			post_weights[stimulus] += SE.w[synapse]

subplot(414)
plot(range(0, stimulus_count), pre_weights, '.b', label = 'Prior Weights')
plot(range(0, stimulus_count), post_weights, '.r', label = 'Post Weights')
xlabel('Stimulus Number')
ylabel('Incoming Excitatory Weight')
legend()

show()
