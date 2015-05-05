# -*- coding: utf-8 -*-
from brian2 import *
from scipy import signal
from lif import *
from syn import *

n = 1000
ne = 800

prefs.codegen.target = 'weave'
defaultclock.dt = 1*ms
duration = 1000*ms

N = LifNeurons(n)

SE = ExcitatorySynapses(N)
SE.connect('i < ne and i != j', p = 0.1)
SE.w = 'rand() * 0.5'
source = SE.i[0]
SE.w[:,source] = 0

SI = InhibitorySynapses(N)
SI.connect('i >= ne and i != j', p = 0.1)
SI.w = -1
SI.w[:,source] = 0
target = SE.j[0]

input_model = """
dr/dt = (-r + rand())/dt : 1
I = (eq_exc - v_reset) * tau_mem/dt * (r * 0.05 * int(i != source) + 0.02 * int(i == source)) : 1
"""
IN = NeuronGroup(N.N, input_model)
N.I = linked_var(IN, 'I')

state_monitor = StateMonitor(N, 'v', record = [source, target])
spike_monitor = SpikeMonitor(N)

network = Network()
network.add(N, IN, SE, SI, state_monitor, spike_monitor)

for sim in range(0, 11):
	SE.w[0] = sim * 1.0
	network.run(duration, report = 'stdout')

figure()

subplot(311)
plot(state_monitor.t/ms, state_monitor.v[0], label = 'v0')
plot(state_monitor.t/ms, state_monitor.v[1], label = 'v1')
legend()
xlabel('Time (ms)')
ylabel('Activation')

subplot(312)
counts = zeros((2, 11))
for i, t in zip(spike_monitor.i, spike_monitor.t):
	if i == source:
		counts[0, int(t / second)] += 1
	elif i == target:
		counts[1, int(t / second)] += 1
plot(range(0, 11), counts[0], label = 'Neuron 0')
plot(range(0, 11), counts[1], label = 'Neuron 1')
legend()
xlabel('Time (ms)')
ylabel('Spike Count')

subplot(313)
plot(spike_monitor.t/ms, spike_monitor.i, ',k')
text(10, 10, "Firing Rate: {0} Hz".format(spike_monitor.num_spikes / (10.0 * duration * n)))
xlabel('Time (ms)')
ylabel('Neuron Index')

show()
