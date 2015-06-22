# -*- coding: utf-8 -*-
from lif import *
from syn import *

n = 1000
ne = 800

prefs.codegen.target = 'weave'
defaultclock.dt = 1*ms
duration = 1000*ms

figure(figsize=(12,16))

input_values = arange(0.02, 0.2, 0.005)
for connectivity in [0.05, 0.1, 0.15]:
	firing_rates = []
	for input in input_values:
		N = LifNeurons(n)
		IN = NoiseInput(N, input)
		
		SE = ExcitatorySynapses(N)
		SE.connect('i < ne and i != j', p = connectivity)
		SE.w = "rand() * 0.5"
		
		SI = InhibitorySynapses(N)
		SI.connect('i >= ne and i != j', p = connectivity)
		SI.w = -1.0
		
		spike_monitor = SpikeMonitor(N)
		
		network = Network()
		network.add(N, IN, SE, SI, spike_monitor)
		network.run(duration, report = 'stdout')
		
		firing_rates.append(spike_monitor.num_spikes / (duration * float(n)))
	
	plot(input_values, firing_rates, label='connectivity={0}'.format(connectivity))

yscale('log')
xlabel('Input')
ylabel('Firing Rate (Hz)')
show()
