# -*- coding: utf-8 -*-
from lif import *
from syn import *

n = 1000
ne = 800

prefs.codegen.target = 'weave'
defaultclock.dt = 1*ms
duration = 1000*ms

figure(figsize=(12,16))
for sim in range(0, 5):
	N = LifNeurons(n)
	IN = NoiseInput(N, 0.05)
	
	SE = ExcitatorySynapses(N)
	SE.connect('i < ne and i != j', p = 0.1)
	SE.w = "rand() * %f" % (sim * 0.25)
	
	SI = InhibitorySynapses(N)
	SI.connect('i >= ne and i != j', p = 0.1)
	SI.w = -1.0
	
	spike_monitor = SpikeMonitor(N)
	state_monitor = StateMonitor(N, 'v', record = [0, 1, 2, 3, 4])
	
	network = Network()
	network.add(N, IN, SE, SI, spike_monitor, state_monitor)
	network.run(duration, report = 'stdout')
	
	subplot(5, 2, 2 * sim + 1)
	plot(spike_monitor.t/ms, spike_monitor.i, ',k')
	text(10, 10, "Firing Rate: {0} Hz".format(spike_monitor.num_spikes / (duration * float(n))))
	xlabel('Time (ms)')
	ylabel('Neuron index')
	
	subplot(5, 2, 2 * sim + 2)
	v = state_monitor.v
	for spike in zip(spike_monitor.i, spike_monitor.t):
		if (0 <= spike[0] < 5):
			v[spike[0], int(spike[1] / defaultclock.dt)] = v_peak
	plot(state_monitor.t/ms, v[0])
	plot(state_monitor.t/ms, v[1])
	plot(state_monitor.t/ms, v[2])
	plot(state_monitor.t/ms, v[3])
	plot(state_monitor.t/ms, v[4])
	xlabel('Time (ms)')
	ylabel('Potential (mV)')

show()
