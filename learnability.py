#!/usr/bin/env python
# -*- coding: utf-8 -*-
from lif import *
from syn import *
from da_stdp import *

def setup_network(params):
    prefs.codegen.target = 'weave'
    defaultclock.dt = 1.0*ms
    neurons = LifNeurons(1000, params)
    excitatory_synapses = DaStdpSynapses(neurons[:800], neurons, params)
    inhibitory_synapses = InhibitorySynapses(neurons[800:], neurons, params)
    network = Network()
    network.add(neurons, excitatory_synapses, inhibitory_synapses)
    return neurons, excitatory_synapses, inhibitory_synapses, network

def run_sim(name, index, duration, neurons, excitatory_synapses, network, params):
    print("Simulation Started: {0} {1}".format(name, index))
    rate_monitor = PopulationRateMonitor(neurons)
    state_monitor = StateMonitor(excitatory_synapses, ('w', 'r'), record=range(5))
    network.add(rate_monitor, state_monitor)
    periods = int(duration / 60*second)
    spikes = ndarray((0, 2))
    from timeit import default_timer
    real_start_time = default_timer()
    weights = ndarray((periods + 1, size(excitatory_synapses.w)))
    weights[0] = excitatory_synapses.w
    for period in range(1, periods + 1):
        spike_monitor = SpikeMonitor(neurons)
        network.add(spike_monitor)
        network.run(1*second)
        spikes = append(spikes, array((spike_monitor.t, spike_monitor.i)).T, axis=0)
        network.remove(spike_monitor)
        network.run(59*second)
        real_elapsed_time = default_timer() - real_start_time
        print("Simulation Status: {0} {1} - {2:%} in {3} sec.".format(
              name, index, float(period) / periods, real_elapsed_time))
        weights[period] = excitatory_synapses.w
    filename = "{0}_{1}".format(name, index)
    numpy.savez_compressed(filename, duration=duration, params=params,
                           t=rate_monitor.t, rate=rate_monitor.rate,
                           w5=state_monitor.w, r=state_monitor.r[0],
                           weights=weights, spikes=spikes)
    print("Simulation Ended: {0} {1}".format(name, index))

class Unpacker(object):
    def __init__(self, fn):
        self.fn = fn
    def __call__(self, packed):
        print "unpacked: {0}".format(packed)
        self.fn(*packed)

def run_parallel(fn, inputs):
    unpacker = Unpacker(fn)
    from multiprocessing import Pool
    pool = Pool()
    pool.map(unpacker, inputs)
    pool.close()
    pool.join()
