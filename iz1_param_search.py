#!/usr/bin/env python
# -*- coding: utf-8 -*-
from learn_iz1 import *
from itertools import product, imap, izip

if __name__ == "__main__":
    duration = 3600*second
    input_levels = [0.025, 0.05, 0.1]
    connectivity_levels = [0.05, 0.1, 0.2]
    reward_levels = [0.25, 0.5, 1.0]
    num_simulations = len(input_levels) * len(connectivity_levels) * len(reward_levels)
    levels = product(input_levels, connectivity_levels, reward_levels)
    inputs = imap(lambda i,x: ("learn_iz1/initial_{0}".format(i), duration,
                               x[0], x[1], x[2]), range(0, num_simulations), levels)
    run_sim.parallel(inputs)
    #for e in inputs:
        #print("Running Simulation {0}: {1}".format(simulation_number, e))
