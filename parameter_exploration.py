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

with open("DK_SC/DK_SC.pkl", "rb") as f:
    sc = pickle.load(f)

labels = sc.region_labels
to_delete_ctx = []
for i, lab in enumerate(labels):
    if not lab.startswith('ctx-'):
        to_delete_ctx.append(i)

labels = np.delete(labels, to_delete_ctx)
sc.region_labels = labels
weights = sc.weights
weights = np.delete(weights, to_delete_ctx, 0)
weights = np.delete(weights, to_delete_ctx, 1)
sc.weights = weights
tls = sc.tract_lengths
tls = np.delete(tls, to_delete_ctx, 0)
tls = np.delete(tls, to_delete_ctx, 1)
sc.tract_lengths = tls
sc.configure()

model = models.ReducedWongWangExcInh()

sim = simulator.Simulator(
    model=model,
    connectivity=sc,
    coupling=coupling.Linear(a=np.array([1.])),
    integrator=integrators.HeunStochastic(dt=1),
    monitors=(monitors.Raw(), monitors.Bold(period=500)),
    simulation_length=60e3
).configure()

def explore(G, sigma, tau_e, tau_i, simulation_length=585e3, sim_dt=0.5, bold_period=2250, offset_time=60e3):

    sim.model.G = np.array([G])
    sim.model.tau_e = np.array([tau_e])
    sim.model.tau_i = np.array([tau_i])
    sim.integrator.noise = noise.Additive(nsig=np.array([sigma]))

    sim.simulation_length = simulation_length
    sim.integrator.dt = sim_dt
    sim.monitors[1].period = bold_period

    sim.configure()

    (rawt, rawd), (boldt, boldd) = sim.run()

    offset = int(offset_time // bold_period)
    ts = boldd[offset:, 0, :, 0]
    
    # to_remove = []
    # for i, lab in enumerate(sc.region_labels): # TODO: find out why it matters that its removed here and not afterwards
    #     if "caudalmiddlefrontal" in lab:
    #         to_remove.append(i)
    
    # ts = np.delete(ts, to_remove, axis=1)

    fc = np.corrcoef(ts, rowvar=False)
    fc[fc == 1] = 0.9999999

    return fc