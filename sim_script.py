#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import numpy as np
from tvb.simulator.lab import * 
from tvb.basic.neotraits.api import NArray, List, Range, Final
import pandas as pd
from timeit import default_timer as timer
from parameter_exploration import explore

def get_config():
    print('get_config')
    parser = argparse.ArgumentParser()

    cmd_parameters = list()
    cmd_parameters.append(["name", "", str])
    #model params
    cmd_parameters.append(["G", 1.0, float])
    cmd_parameters.append(["sigma", 0.01, float])
    cmd_parameters.append(["tau_e", 100.0, float])

    cmd_parameters.append(["sim_length", 600e3, float]) 
    cmd_parameters.append(["sim_dt", 0.5, float])
    cmd_parameters.append(["bold_period", 2250, float])
    cmd_parameters.append(["offset_time", 60e3, float])

    for (parname, default, partype) in cmd_parameters:
        parser.add_argument(f"-{parname}", default=default, type=partype)
    config = parser.parse_args()
    return config

def run_simulation(config):
    tstart = timer()
    print("Hi! I'm a job with the following commandline parameters:")
    print(f"{config.__dict__}")
    print("And these are the important simulation parameters:")
    print(f"G = {config.G}")
    print(f"sigma = {config.sigma}")
    print(f"tau_e = {config.tau_e}")
    print(f"Time is: {tstart}")

    # simulation code goes here
    # print(f'run_simulation({config.G}, {config.sigma})')
    # part_fname = f"G_{config.G}_sigma_{config.sigma}"
    print(f'run_simulation({config.tau_e})')
    part_fname = f"subid_tau_e_{config.tau_e}"

    fc = explore(
        config.G, config.sigma, config.tau_e,
        config.sim_length, config.sim_dt, 
        config.bold_period, config.offset_time
    )

    res = {
        "G": config.G,
        "sigma": config.sigma,
        "tau_e": config.tau_e,
        "FC": fc
    }

    print(f'saving: RWW_result_{part_fname}')
    os.makedirs(f'out', exist_ok=True)
    with open(f'out/RWW_result_{part_fname}.pkl', 'wb') as f:
        pickle.dump(res, f)

    print(f"Simulation finished in {timer() - tstart} seconds")


# The if statement prevents running the simulation when importing this scipt into some other script
# For example: you might want to re-use generate_filename() in order to load the data systematically
# and plot something
# Something like: from simulation import generate_filename
if (__name__ == "__main__"):
    config = get_config()
    
    run_simulation(config)
