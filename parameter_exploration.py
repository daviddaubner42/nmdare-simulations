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

    labels = sc.region_labels
    labels = np.delete(labels, to_remove)
    for i, lab in enumerate(labels):
        if "ctx" in lab:
            _, hemi, name = lab.split('-')
            labels[i] = f"{hemi}_{name}"
        else:
            labels[i] = lab.upper()
        if "THALAMUS" in labels[i]:
            hemi, name = labels[i].split('-')
            labels[i] = f"{hemi}-{name}-PROPER"
            
    with open("resources/network_labels.pkl", "rb") as f:
        network_labels = pickle.load(f)
    to_delete_nl = []
    for i, lab in enumerate(network_labels):
        if "VENTRALDC" in lab:
            to_delete_nl.append(i)
    network_labels = np.delete(network_labels, to_delete_nl)
    transform = {}
    for i, lab in enumerate(network_labels):
        transform[i] = np.where(labels == lab)
    new_ts = np.zeros_like(ts)
    for i in range(len(labels)):
        new_ts[:, i] = ts[:, transform[i]].squeeze()
    ts = new_ts

    fc = np.corrcoef(ts, rowvar=False)
    fc[fc == 1] = 0.9999999

    return FCuCorrelation(fc, target_fc)