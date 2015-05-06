# -*- coding: utf-8 -*-
import random as rnd
from scipy import signal
from lif import *
from syn import *
from stdp import *

timestep = 1.0*ms
neuron_count = 200
excitatory_count = 160
input_rate = 0.025
minimum_weight = 0.0
maximum_weight = 0.5
inhibitory_weight = -0.5
connectivity = 0.1

stimulus_size = 25
stimulus_count = 10
minimum_isi = 100*ms
maximum_isi = 300*ms

prefs.codegen.target = 'weave'
defaultclock.dt = timestep

N = LifNeurons(neuron_count)
stimulus_array = zeros((stimulus_count, neuron_count))
for x in range(0,stimulus_count):
	for y in rnd.sample(range(0, excitatory_count), stimulus_size):
		stimulus_array[x,y] = 1
stimuli = TimedArray(stimulus_array, dt = 1*ms)

input_model = """
stimulus : second (shared)
t_stimulus : second (shared)
dr/dt = (-r + rand())/dt : 1
stim = int(t >= t_stimulus) * stimuli(stimulus, i) * 0.1 : 1
I = (eq_exc - v_reset) * tau_mem/ms * r * (stim + %f) : 1
""" % input_rate
IN = NeuronGroup(neuron_count, model = input_model)
IN.stimulus = 0*ms
IN.t_stimulus = minimum_isi
N.I = linked_var(IN, 'I')

pairings = empty(0)
@network_operation(dt = 1*ms, when = "resets")
def next_stimulus():
	global pairings
	if (IN.t > IN.t_stimulus + 10*ms):
		if (IN.stimulus == 0*ms):
			IN.stimulus = 1*ms
			IN.t_stimulus = IN.t + 20*ms
			pairings = append(pairings, IN.t/second)
		else:
			IN.stimulus = floor(random() * stimulus_count)*ms
			IN.t_stimulus = IN.t + minimum_isi + (maximum_isi - minimum_isi) * random()

SE = StdpSynapses(N)
SE.connect('i < excitatory_count and i != j', p = connectivity)
SE.w = 'minimum_weight + rand() * (maximum_weight - minimum_weight)'
SI = InhibitorySynapses(N)
SI.connect('i >= excitatory_count and i != j', p = connectivity)
SI.w = inhibitory_weight

rate_monitor = PopulationRateMonitor(N)
spike_monitor = SpikeMonitor(N)
state_monitor = StateMonitor(SE, 'w', record = True)

network = Network()
network.add(N, IN, SE, SI, next_stimulus, rate_monitor, spike_monitor, state_monitor)

pre_weights = zeros(stimulus_count)
for stimulus in range(0, stimulus_count):
	for synapse in range(0, asarray(SE.j).size):
		if stimulus_array[stimulus,SE.j[synapse]] == 1:
			pre_weights[stimulus] += SE.w[synapse]

network.run(10*second)
figure()
subplot(411)
plot(spike_monitor.t/second, spike_monitor.i, ',k')
plot(pairings, ones_like(pairings) * -10, 'ob')

network.remove(spike_monitor)
network.run(300*second, report = 'stdout', report_period = 10*second)

spike_monitor = SpikeMonitor(N)
network.add(spike_monitor)
network.run(10*second)
subplot(412)
plot(spike_monitor.t/second, spike_monitor.i, ',k')
pairings = pairings[pairings >= 310]
plot(pairings, ones_like(pairings) * -10, 'ob')

subplot(413)
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
