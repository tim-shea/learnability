# -*- coding: utf-8 -*-
from brian2 import *

tau_exc = 10*ms
tau_inh = 100*ms
p_cb = 0.001
tau_pos = 20*ms
tau_neg = 20*ms
a_pos = 0.01
a_neg = -a_pos * tau_pos / tau_neg * 1.05
w_max = 1.0

def CriticalBranchingSynapses(N, dt = None):
	update_model = """
	w : 1
	enabled : 1
	credited : 1
	dge/dt = -ge / tau_exc : 1
	ge_total_post = ge : 1 (summed)
	dstdp_pos/dt = -stdp_pos / tau_pos : 1
	dstdp_neg/dt = -stdp_neg / tau_neg : 1
	"""
	pre_model = """
	stdp_pos += a_pos
	w = clip(w + stdp_neg, 0, w_max)
	ge = w * enabled
	enabled = clip(enabled + (1 - credited) * int(rand() < p_cb), 0, 1) # Enable uncredited, disabled axonal synapse
	credited = clip(credited * (enabled + (1 - int(rand() < p_cb))), 0, 1) # Clear credit
	"""
	post_model = """
	stdp_neg += a_neg
	w = clip(w + stdp_pos, 0, w_max)
	ge = 0
	enabled = clip(enabled * ((1 - credited) + (1 - int(rand() < p_cb))), 0, 1) # Disable credited, enabled dendritic synapses
	credited = clip(credited + enabled * int(rand() < p_cb), 0, 1) # Assign credit to enabled dendritic synapse
	"""
	S = Synapses(N, model = update_model, pre = pre_model, post = post_model)
	return S

def InhibitoryCBSynapses(N, dt = None):
	update_model = """
	w : 1
	enabled : 1
	credited : 1
	dgi/dt = -gi / tau_exc : 1
	gi_total_post = gi : 1 (summed)
	"""
	pre_model = """
	gi = w * enabled
	enabled = clip(enabled * ((1 - credited) + (1 - int(rand() < p_cb))), 0, 1) # Disable credited, enabled dendritic synapses
	credited = clip(credited + enabled * int(rand() < p_cb), 0, 1) # Assign credit to enabled dendritic synapse
	"""
	post_model = """
	ge = 0
	enabled = clip(enabled + (1 - credited) * int(rand() < p_cb), 0, 1) # Enable uncredited, disabled axonal synapse
	credited = clip(credited * (enabled + (1 - int(rand() < p_cb))), 0, 1) # Clear credit
	"""
	S = Synapses(N, model = update_model, pre = pre_model, post = post_model)
	return S

if __name__ == "__main__":						# Run a simulation to visualize the interaction between CB and STDP
	from lif import *
	from scipy import signal
	
	prefs.codegen.target = 'weave'
	defaultclock.dt = 1*ms
	plot_duration = 10*second
	run_duration = 300*second
	
	figure()
	
	n = 500
	ne = 400
	m = 20
	
	N = LifNeurons(n + m)						# Create hidden and motor neurons
	IN = NoiseInput(N, "int(i < n) * 0.025")	# Only send input to hidden neurons
	SE = CriticalBranchingSynapses(N)			# Create recurrent and forward excitatory CB+STDP synapses
	SE.connect("i < ne and i != j", p = 0.1)
	SE.w = "rand() * 0.5"
	SE.enabled = 0
	SE.credited = 0
	SI = InhibitoryCBSynapses(N)				# Create recurrent and forward inhibitory CB synapses
	SI.connect("i >= ne and i < n and i != j", p = 0.1)
	SI.w = -0.5
	SI.enabled = 1								# Inhibitory synapses begin enabled
	SI.credited = 0
	
	print("Excitatory: %d / %d" % (count_nonzero(SE.enabled), asarray(SE.enabled).size))
	print("Inhibitory: %d / %d" % (count_nonzero(SI.enabled), asarray(SI.enabled).size))
	
	motor_projections = []						# Create the monitors and network and run for initial period
	for i,j in enumerate(SE.j):
		if j >= n:
			motor_projections.append(i)
	state_monitor = StateMonitor(SE, ('w', 'enabled'), motor_projections)
	spike_monitor = SpikeMonitor(N)
	rate_monitor = PopulationRateMonitor(N)
	network = Network()
	network.add(N, IN, SE, SI, state_monitor, spike_monitor, rate_monitor)
	network.run(plot_duration, report = 'stdout')
	network.remove(spike_monitor)
	
	print("Excitatory: %d / %d" % (count_nonzero(SE.enabled), asarray(SE.enabled).size))
	print("Inhibitory: %d / %d" % (count_nonzero(SI.enabled), asarray(SI.enabled).size))
	
	subplot(411)	# Subplot 1: Initial period spike raster
	plot(spike_monitor.t/second, spike_monitor.i, ',k')
	
	# Run the full simulation
	network.run(run_duration, report = 'stdout', report_period = 10*second)
	
	print("Excitatory: %d / %d" % (count_nonzero(SE.enabled), asarray(SE.enabled).size))
	print("Inhibitory: %d / %d" % (count_nonzero(SI.enabled), asarray(SI.enabled).size))
	
	spike_monitor = SpikeMonitor(N)				# Create a new spike monitor and run for final period
	network.add(spike_monitor)
	network.run(plot_duration, report = 'stdout')
	
	print("Excitatory: %d / %d" % (count_nonzero(SE.enabled), asarray(SE.enabled).size))
	print("Inhibitory: %d / %d" % (count_nonzero(SI.enabled), asarray(SI.enabled).size))
	
	subplot(412)	# Subplot 2: Final period spike raster
	plot(spike_monitor.t/second, spike_monitor.i, ',k')
	
	subplot(413)	# Subplot 3: Network spike rate in 1 sec bins
	rates = scipy.stats.binned_statistic(rate_monitor.t/second, rate_monitor.rate, bins = 320)
	plot(rates[1][1:], rates[0])
	
	subplot(414)	# Subplot 4: Summed incoming enabled weight for motor projection synapses
	plot(state_monitor.t/second, sum(multiply(state_monitor.w, state_monitor.enabled), axis = 0))
	
	show()
