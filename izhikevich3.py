#!/usr/bin/env python
# -*- coding: utf-8 -*-
from brian2 import *
from lif import *
from syn import *
from da_stdp import *

def run(name, index, duration, constant_input, connectivity, reward_delay,
        reward_amount, reward_decay, stdp_bias):
    set_device('cpp_standalone')
    defaultclock.dt = 1.0*ms
    par = LifParams(constant_input=constant_input)
    par.update(SynParams())
    par.update(DaStdpParams(a_neg=-1.0 + stdp_bias, connectivity=connectivity))
    par.update({'duration': duration, 'reward_delay': reward_delay,
                'reward_amount': reward_amount, 'tau_da': reward_decay})
    
    n = LifNeurons(1000, par)
    se = DaStdpSynapses(n, par)
    se.connect('i != j and i < 800', p='connectivity')
    se.w = 'w_exc'
    se.delay = 'delay'
    si = InhibitorySynapses(n, par)
    si.connect('i != j and i >= 800', p='connectivity')
    si.w = 'w_inh'
    si.delay = 'delay'
    net = Network()
    net.add(n, se, si)
    
    stim = NeuronGroup(1, model="dtimer/dt = 1 : second",
                       threshold="timer > 1*second", reset="timer = 0*second", namespace=par)
    stim_syn = Synapses(stim, n, namespace=par, model="""
                        tstimulus : second
                        input_post = constant_input * (1 + 5 * int(j < 50 and t - tstimulus < 20*ms)) : 1 (summed)
                        """, pre="tstimulus = t", connect=True)
    stim_syn.tstimulus = -1*second
    resp = NeuronGroup(1, namespace=par, threshold="timer < -reward_time", reset="""
                       ready = 0
                       timer = 0*ms
                       """, model="""
                       ready : 1
                       dtimer/dt = -1 * ready : second
                       response_one : 1
                       response_two : 1
                       response_rate = (response_one / response_two) : 1
                       reward_time = 1000*ms * int(response_rate > 1) + (reward_delay / response_rate) : second
                       """)
    resp.ready = 0
    resp.timer = 0*ms
    resp.response_one = 1
    resp.response_two = 1
    sr_syn = Synapses(stim, resp, connect=True, pre="""
                      ready = 1
                      timer_post = 20*ms
                      response_one = 1
                      response_two = 1
                      """)
    sr_syn.delay = 20*ms
    resp_syn = Synapses(n, resp, connect=True, namespace=par, pre="""
                        response_one += 1 * int(i >= 50 and i < 100) * int(timer > 0*ms)
                        response_two += 1 * int(i >= 100 and i < 150) * int(timer > 0*ms)
                        """, post="da_pre += reward_amount")
    net.add(stim, stim_syn, resp, sr_syn, resp_syn)
    
    rate = PopulationRateMonitor(n)
    spikes = SpikeMonitor(n)
    state = StateMonitor(resp, ('ready', 'timer', 'response_one', 'response_two', 'reward_time'), record=True)
    state2 = StateMonitor(n, 'da', record=[0])
    net.add(rate, spikes, state, state2)
    
    net.run(duration, report='stdout', report_period=60*second, namespace={})
    device.build(directory='output', compile=True, run=True, debug=False)
    syn = array((se.i, se.j, se.w))
    spk = array((spikes.t, spikes.i))
    filename = "iz3_{0}_{1}".format(name, index)
    numpy.savez_compressed(filename, params=par, t=rate.t,
                           rate=rate.rate, syn=syn, spk=spk,
                           ready=state.ready[0], timer=state.timer[0],
                           resp1=state.response_one[0], resp2=state.response_two[0],
                           rew_t=state.reward_time[0], da=state2.da[0])

if __name__ == "__main__":
    if len(sys.argv) == 10:
        name = sys.argv[1]
        index = int(sys.argv[2])
        duration = float(sys.argv[3]) * second
        input_rate = float(sys.argv[4])
        connectivity = float(sys.argv[5])
        reward_delay = float(sys.argv[6]) * ms
        reward_amount = float(sys.argv[7])
        reward_decay = float(sys.argv[8]) * ms
        stdp_bias = float(sys.argv[9])
        run(name, index, duration, input_rate, connectivity, reward_delay,
            reward_amount, reward_decay, stdp_bias)
