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

# with open("TVB_input/tumor_SC.pkl", "rb") as f:
#     sc = pickle.load(f)

with open("DK_SC/DK_SC.pkl", "rb") as f:
    sc = pickle.load(f)

labels = sc.region_labels
to_delete_ctx = []
for i, lab in enumerate(labels):
    if not lab.startswith('ctx-'):
        to_delete_ctx.append(i)

# labels = np.delete(labels, to_delete_ctx)
# sc.region_labels = labels
# weights = sc.weights
# weights = np.delete(weights, to_delete_ctx, 0)
# weights = np.delete(weights, to_delete_ctx, 1)
# sc.weights = weights
# tls = sc.tract_lengths
# tls = np.delete(tls, to_delete_ctx, 0)
# tls = np.delete(tls, to_delete_ctx, 1)
# sc.tract_lengths = tls
# sc.configure()

model = models.ReducedWongWangExcInh()

sim = simulator.Simulator(
    model=model,
    connectivity=sc,
    coupling=coupling.Linear(a=np.array([1.])),
    integrator=integrators.HeunStochastic(dt=1),
    monitors=(monitors.Raw(), monitors.Bold(period=500)),
    simulation_length=60e3
).configure()

# target_fc = np.loadtxt("FCs/avg_hc_fc.csv", delimiter=',')
# target_fc = target_fc[:66, :66]

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
    
    # ----------------------------------------------
    labels = sc.region_labels

    new_labels = labels.tolist()
    for i, lab in enumerate(new_labels):
        if "ctx" in lab:
            _, hemi, name = lab.split('-')
            new_labels[i] = f"{hemi}_{name}"
        else:
            new_labels[i] = lab.upper()
    new_labels = np.array(new_labels)

    to_delete_sc = []
    for i, lab in enumerate(new_labels):
        if "caudalmiddlefrontal" in lab:
            to_delete_sc.append(i)

    ts = np.delete(ts, to_delete_sc, 1)
    new_labels = np.delete(new_labels, to_delete_sc)

    with open("resources/network_labels.pkl", "rb") as f:
        network_labels = pickle.load(f)
    to_delete_fc = []
    for i, lab in enumerate(network_labels):
        if not lab in new_labels:
            to_delete_fc.append(i)
    network_labels = np.delete(network_labels, to_delete_fc)

    transform = {}
    for i, lab in enumerate(network_labels):
        transform[i] = np.where(new_labels == lab)

    new_ts = np.zeros_like(ts)
    for i in range(len(new_labels)):
        new_ts[:, i] = ts[:, transform[i]].squeeze()
    ts = new_ts

    # ------------------------------------------------------

    fc = np.corrcoef(ts, rowvar=False)
    fc[fc == 1] = 0.9999999

    return fc