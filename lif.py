# -*- coding: utf-8 -*-
from matplotlib.pyplot import *
from brian2 import *

tau_mem = 50*ms
v_threshold = 1
v_peak = 1.5
v_reset = 0
eq_leak = 0
eq_exc = 1.5
eq_inh = 0
refractory_period = 5*ms

def LifNeurons(n, dt = None):
	neuron_model = """
	I : 1 (linked)
	ge_total : 1
	gi_total : 1
	dv/dt = (-(v - eq_leak) + (eq_exc - v) * ge_total + (v - eq_inh) * gi_total + I) / tau_mem : 1 (unless refractory)
	"""
	reset_model = "v = v_reset"
	N = NeuronGroup(n, neuron_model, threshold = 'v > v_threshold', reset = reset_model, refractory = refractory_period)
	N.v = 'v_reset + (v_threshold - v_reset) * rand()'
	return N

def NoiseInput(N, x, dt = None):
	input_model = """
	dr/dt = (-r + rand())/dt : 1
	I = (eq_exc - v_reset) * tau_mem/ms * (%s) * r : 1
	""" % x
	IN = NeuronGroup(N.N, input_model, dt = dt)
	N.I = linked_var(IN, 'I')
	return IN

def BernoulliSpikeInput(N, p, dt = None):
	input_model = """
	dr/dt = (-r + rand())/dt : 1
	I = (eq_exc - v_reset) * tau_mem/dt * int(r < (%s)) : 1
	""" % p
	IN = NeuronGroup(N.N, input_model, dt = dt)
	N.I = linked_var(IN, 'I')
	return IN

if __name__ == "__main__":
	prefs.codegen.target = 'numpy'
	defaultclock.dt = 1*ms
	duration = 1*second
	
	N = LifNeurons(1)
	IN = NoiseInput(N, 0.05)
	spike_monitor = SpikeMonitor(N)
	state_monitor = StateMonitor(N, ('v', 'I'), record = True, when = 'thresholds')
	network = Network()
	network.add(N, IN, spike_monitor, state_monitor)
	network.run(duration, report = 'stdout')
	
	figure()
	
	subplot(211)
	plot(state_monitor.t/second, 0.1 * state_monitor.I[0])
	v = state_monitor.v[0]
	for t_spike in spike_monitor.t:
		i_spike = int(t_spike / defaultclock.dt)
		v[i_spike] = v_peak
	plot(state_monitor.t/second, v)
	xlabel('Time (s)')
	ylabel('Input Current')
	
	N = LifNeurons(1)
	IN = BernoulliSpikeInput(N, 0.01)
	spike_monitor = SpikeMonitor(N)
	state_monitor = StateMonitor(N, ('v', 'I'), record = True, when = 'thresholds')
	network = Network()
	network.add(N, IN, spike_monitor, state_monitor)
	network.run(duration, report = 'stdout')
	
	subplot(212)
	plot(state_monitor.t/second, 0.1 * state_monitor.I[0])
	v = state_monitor.v[0]
	for t_spike in spike_monitor.t:
		i_spike = int(t_spike / defaultclock.dt)
		v[i_spike] = v_peak
	plot(state_monitor.t/second, v)
	xlabel('Time (s)')
	ylabel('Input Current')
	
	show()
