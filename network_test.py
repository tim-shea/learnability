#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scipy.stats import binned_statistic as bin_stat
from lif import *
from syn import *

prefs.codegen.target = 'numpy'
defaultclock.dt = 1*ms

params = LifParams(constant_input=3)
params.update(SynParams())
neurons = LifNeurons(1000, params)
excitatory_synapses = ExcitatorySynapses(neurons, params)
excitatory_synapses.connect('i != j and i < 800', p=0.1)
excitatory_synapses.w = 1.0
inhibitory_synapses = InhibitorySynapses(neurons, params)
inhibitory_synapses.connect('i != j and i >= 800', p=0.1)
inhibitory_synapses.w = -1.0
rate_monitor = PopulationRateMonitor(neurons)
spike_monitor = SpikeMonitor(neurons)
network = Network()
network.add(neurons, excitatory_synapses, inhibitory_synapses, rate_monitor, spike_monitor)
network.run(10*second, report='stdout', report_period=1.0*second, namespace={})

figure()
subplot(211)
suptitle('Network Activity')
binned_rate = bin_stat(rate_monitor.t/second, rate_monitor.rate, bins=100)
plot(binned_rate[1][:-1], binned_rate[0])
ylabel('Firing Rate (Hz)')
subplot(212)
plot(spike_monitor.t/second, spike_monitor.i, '.k')
ylabel('Neuron #')
xlabel('Time (s)')
show()
