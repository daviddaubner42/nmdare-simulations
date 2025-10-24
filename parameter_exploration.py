import matplotlib.pyplot as plt
import numpy as np
from tvb.simulator.lab import *
import pickle
import os
import pandas as pd

def FCuCorrelation(FC1, FC2, fisher=True):
    u_idx = np.triu_indices_from(FC1, k=1)
    FCu1 = FC1.copy()[u_idx]
    FCu2 = FC2.copy()[u_idx]
    if fisher:
        FCu1 = np.arctanh(FCu1)
        FCu2 = np.arctanh(FCu2)
    FC_corr_upper = np.corrcoef(FCu1, FCu2)[0, 1]
    return FC_corr_upper

with open("TVB_input/tumor_SC.pkl", "rb") as f:
    sc = pickle.load(f)

# with open("DK_SC/DK_SC.pkl", "rb") as f:
#     sc = pickle.load(f)
sc.configure()

model = models.ReducedWongWangExcInh()

sim = simulator.Simulator(
    model=model,
    connectivity=sc,
    coupling=coupling.Linear(a=np.array([1.])),
    integrator=integrators.HeunStochastic(dt=0.5),
    monitors=(monitors.Raw(), monitors.Bold(period=2250)),
    simulation_length=300e3
).configure()

target_fc = np.loadtxt("FCs/avg_hc_fc.csv", delimiter=',')
target_fc = target_fc[:66, :66]

def explore(G, sigma, simulation_length=585e3, sim_dt=0.5, bold_period=2250, offset_time=60e3):

    sim.model.G = np.array([G])
    sim.integrator.noise = noise.Additive(nsig=np.array([sigma]))

    sim.simulation_length = simulation_length
    sim.integrator.dt = sim_dt
    sim.monitors[1].period = bold_period

    sim.configure()

    (rawt, rawd), (boldt, boldd) = sim.run()

    offset = int(offset_time // bold_period)
    ts = boldd[offset:, 0, :, 0]

    to_remove = []
    for i, lab in enumerate(sc.region_labels):
        if "caudalmiddlefrontal" in lab:
            to_remove.append(i)

    ts = np.delete(ts, to_remove, axis=1)

    fc = np.corrcoef(ts, rowvar=False)
    fc[fc == 1] = 0.9999999

    return FCuCorrelation(fc, target_fc)