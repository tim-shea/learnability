#!/usr/bin/env python
# -*- coding: utf-8 -*-
from learn_iz1 import *

if __name__ == "__main__":
    duration = 10*second
    input_values = [0.025, 0.05, 0.1]
    connectivity_values = [0.05, 0.1, 0.2]
    reward_values = [0.25, 0.5, 1.0]
    experiment_directory = "learn_iz1"
    experiment_id = "learn_iz1/TEST_{0}"
    simulation_number = 0
    for i in input_values:
        for c in connectivity_values:
            for r in reward_values:
                run_sim(experiment_id.format(simulation_number), duration, i, c, r)
                simulation_number += 1
