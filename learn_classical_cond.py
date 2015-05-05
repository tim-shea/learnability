# -*- coding: utf-8 -*-
from stdp import *
from random import *
from scipy import signal

timestep = 1.0*ms
neuron_count = 40
excitatory_count = 30
input_rate = 0.00
minimum_weight = 0.0
maximum_weight = 1.0
inhibitory_weight = -0.5
connectivity = 0.5

tau_pre = 20*ms
tau_post = 20*ms
tau_elig = 100*ms
a_post = a_pre = 1.0

minimum_isi = 500*ms
maximum_isi = 1000*ms
stimulus_duration = 10*ms

prefs.codegen.target = 'weave'
defaultclock.dt = timestep

N1 = LifNeurons(neuron_count)
N2 = LifNeurons(neuron_count)

input_model = """
t_stimulus : second (shared)
dr/dt = (-r + rand())/dt : 1
inject_spike = int(r < %f + int(t > t_stimulus) * stimuli(stimulus, i) * 0.2) : 1
I = (eq_exc - v_reset) * tau_mem/ms * inject_spike : volt
""" % input_rate
IN1 = NeuronGroup(neuron_count, model = input_model)
IN.stimulus = 0*ms
IN.t_stimulus = minimum_isi
N.I = linked_var(IN, 'I')

targets = empty(0)
rewards = empty(0)
@network_operation(dt = 1*ms)
def next_stimulus():
	global targets, rewards
	if (IN.t > IN.t_stimulus + stimulus_duration):
		if (IN.stimulus == 0*ms):
			IN.reward = IN.t + random() * reward_delay
			targets = append(targets, IN.t/second)
			rewards = append(rewards, IN.reward/second)
		IN.t_stimulus = N.t + minimum_isi + (maximum_isi - minimum_isi) * random()
		IN.stimulus = floor(random() * stimulus_count)*ms

SE = DaStdpSynapses(N)
SE.connect('i < excitatory_count and i != j', p = connectivity)
SE.w = 'minimum_weight + rand() * (maximum_weight - minimum_weight)'
SI = Synapses(N, pre = 'gi_post += inhibitory_weight')
SI.connect('i >= excitatory_count and i != j', p = connectivity)

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
network.run(10*second, report = 'stdout', report_period = 10*second)

spike_monitor = SpikeMonitor(N)
network.add(spike_monitor)
network.run(10*second)
subplot(412)
plot(spike_monitor.t/second, spike_monitor.i, ',k')
clipped_targets = numpy.array(targets)
clipped_targets = clipped_targets[logical_and(clipped_targets >= 910, clipped_targets < 920)]
plot(clipped_targets, ones_like(clipped_targets) * -10, 'ob')
clipped_rewards = numpy.array(rewards)
clipped_rewards = clipped_rewards[logical_and(clipped_rewards >= 910, clipped_rewards < 920)]
plot(clipped_rewards, ones_like(clipped_rewards) * -10, '^r')

subplot(413)
#rates = scipy.stats.binned_statistic(rate_monitor.t/second, rate_monitor.rate, bins = 3000)
#plot(rates[1][1:], rates[0], label = 'Firing Rate')
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
