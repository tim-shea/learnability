# -*- coding: utf-8 -*-
import random as rnd
from scipy import signal
from lif import *
from syn import *
from stdp import *

timestep = 1.0*ms
neuron_count = 1000
excitatory_count = 800
minimum_weight = 0.0
maximum_weight = 0.5
inhibitory_weight = -0.5
connectivity = 0.1

stimulus_size = 50
stimulus_intensity = 0.1
constant_intensity = 0.0
stimulus_duration = 100*ms
test_trials = 10
training_trials = 10
steps_per_interval = 5 # interstimulus interval = 400 ms

prefs.codegen.target = 'numpy'
defaultclock.dt = timestep

N = LifNeurons(neuron_count)

no_stimulus = full(neuron_count, constant_intensity)
unconditioned_stimulus = copy(no_stimulus)
unconditioned_stimulus[0:stimulus_size] = stimulus_intensity
conditioned_stimulus = copy(no_stimulus)
conditioned_stimulus[stimulus_size:2*stimulus_size] = stimulus_intensity
paired_stimulus = copy(no_stimulus)
paired_stimulus[0:2*stimulus_size] = stimulus_intensity

pretest_trial = tile(no_stimulus, (steps_per_interval, 1))
pretest_trial[0] = unconditioned_stimulus
pretest = TimedArray(tile(pretest_trial, (test_trials, 1)), dt=stimulus_duration)

training_trial = tile(no_stimulus, (steps_per_interval, 1))
training_trial[0] = paired_stimulus
training = TimedArray(tile(training_trial, (training_trials, 1)), dt=stimulus_duration)

posttest_trial = tile(no_stimulus, (steps_per_interval, 1))
posttest_trial[0] = conditioned_stimulus
posttest = TimedArray(tile(posttest_trial, (test_trials, 1)), dt=stimulus_duration)


#conditioned_stimulus = arange(0:stimulus_size)
#unconditioned_stimulus = arange(stimulus_size:2 * stimulus_size)
#
#input_model = """
#training : 1 (shared)
#t_stimulus : second (shared)
#dr/dt = (-r + rand())/dt : 1
#stimulation = int(t >= t_stimulus) * (int(i < stimulus_size) + int(i < 2 * stimulus_size) * training) : 1
#intensity = stimulation * stimulus_intensity + constant_intensity : 1
#I = (eq_exc - v_reset) * tau_mem/ms * r * (intensity) : 1
#"""
#IN = NeuronGroup(neuron_count, model = input_model)
#IN.stimulus = 0*ms
#IN.t_stimulus = minimum_isi
#N.I = linked_var(IN, 'I')
#
#pairings = empty(0)
#@network_operation(dt = 1*ms, when = "resets")
#def next_stimulus():
#    global pairings
#    if (IN.t > IN.t_stimulus + 10*ms):
#        if (IN.stimulus == 0*ms):
#            IN.stimulus = 1*ms
#            IN.t_stimulus = IN.t + 20*ms
#            pairings = append(pairings, IN.t/second)
#        else:
#            IN.stimulus = floor(random() * stimulus_count)*ms
#            IN.t_stimulus = IN.t + minimum_isi + (maximum_isi - minimum_isi) * random()
#
#SE = StdpSynapses(N)
#SE.connect('i < excitatory_count and i != j', p = connectivity)
#SE.w = 'minimum_weight + rand() * (maximum_weight - minimum_weight)'
#SI = InhibitorySynapses(N)
#SI.connect('i >= excitatory_count and i != j', p = connectivity)
#SI.w = inhibitory_weight
#
#rate_monitor = PopulationRateMonitor(N)
#spike_monitor = SpikeMonitor(N)
#state_monitor = StateMonitor(SE, 'w', record = True)
#
#network = Network()
#network.add(N, IN, SE, SI, next_stimulus, rate_monitor, spike_monitor, state_monitor)
#
#pre_weights = zeros(stimulus_count)
#for stimulus in range(0, stimulus_count):
#    for synapse in range(0, asarray(SE.j).size):
#        if stimulus_array[stimulus,SE.j[synapse]] == 1:
#            pre_weights[stimulus] += SE.w[synapse]
#
#network.run(10*second)
#figure()
#subplot(411)
#plot(spike_monitor.t/second, spike_monitor.i, ',k')
#plot(pairings, ones_like(pairings) * -10, 'ob')
#
#network.remove(spike_monitor)
#network.run(60*second, report = 'stdout', report_period = 10*second)
#
#spike_monitor = SpikeMonitor(N)
#network.add(spike_monitor)
#network.run(10*second)
#subplot(412)
#plot(spike_monitor.t/second, spike_monitor.i, ',k')
#pairings = pairings[pairings >= 310]
#plot(pairings, ones_like(pairings) * -10, 'ob')
#
#subplot(413)
#plot(state_monitor.t/second, sum(state_monitor.w, axis = 0)/asarray(SE.w).size, label = 'mean weight')
#legend()
#
#post_weights = zeros(stimulus_count)
#for stimulus in range(0, stimulus_count):
#    for synapse in range(0, asarray(SE.j).size):
#        if stimulus_array[stimulus,SE.j[synapse]] == 1:
#            post_weights[stimulus] += SE.w[synapse]
#
#subplot(414)
#plot(range(0, stimulus_count), pre_weights, '.b', label = 'Prior Weights')
#plot(range(0, stimulus_count), post_weights, '.r', label = 'Post Weights')
#xlabel('Stimulus Number')
#ylabel('Incoming Excitatory Weight')
#legend()
#
#show()
