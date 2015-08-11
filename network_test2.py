#!/usr/bin/env python
# -*- coding: utf-8 -*-
from brian2 import *
from lif import *
from syn import *

prefs.codegen.target = 'numpy'
defaultclock.dt = 1.0*ms
params = {'v_rest': 0.0,
          'eq_exc': 1.0,
          'eq_inh': -1.0,
          'tau_mem': 10*ms,
          'tau_ge': 10*ms,
          'tau_gi': 10*ms,
          'v_thresh': 0.95,
          'ref': 5*ms,
          'w_exc': 0.5,
          'w_inh': -0.5}

model = """
exc : 1
ge : 1
gi : 1
I : 1
g = (v_rest - v) + ge * (eq_exc - v) + gi * (v - eq_inh) : 1
dv/dt = (g + I) / tau_mem : 1 (unless refractory)
"""
n = NeuronGroup(1000, model, threshold='v > v_thresh', reset="v = v_rest",
                refractory='ref', namespace=params)
n.v = "v_rest + (v_thresh - v_rest) * rand()"
n.exc = "int(i < 800)"

noise = NeuronGroup(1000, 'input : 1')
noise.input = 0.0
sn = Synapses(noise, n, model='I_post = 1.0 + rand() * input_pre : 1 (summed)', connect='i == j')
se = Synapses(n, namespace=params, delay=1*ms,
              model="""
              dge_syn/dt = -ge_syn / tau_ge : 1
              ge_post = ge_syn : 1 (summed)
              """, pre="ge_syn = w_exc", post="ge_syn = 0")
se.connect('i != j and exc_pre', p=0.1)
si = Synapses(n, namespace=params, delay=1*ms,
              model="""
              dgi_syn/dt = -gi_syn / tau_gi : 1
              gi_post = gi_syn : 1 (summed)
              """, pre="gi_syn = w_inh", post="gi_syn = 0")
si.connect('i != j and not exc_pre', p=0.1)
spikes = SpikeMonitor(n)
state = StateMonitor(n, ('v', 'ge', 'gi', 'I'), record=True)
net = Network()
net.add(n, noise, sn, se, si, spikes, state)
net.run(10*second, report='stdout', report_period=10*second)

figure()
subplot(211)
plot(state.t, state.v[0])
plot(state.t, state.ge[0])
plot(state.t, state.gi[0])
subplot(212)
plot(spikes.t, spikes.i, '.k')
show()
