import matplotlib.pyplot as plt
import numpy as np
from tvb.simulator.lab import *
import pickle
import os
import pandas as pd
from scipy.stats import kstest

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
    monitors=(monitors.Raw(), monitors.Bold(period=2250)),
    simulation_length=585e3
).configure()

# with open("FCDs/hc_hists.csv", "rb") as f:
#     hc_hists = pickle.load(f)
hc_hists = np.loadtxt("FCDs/LEGK010/hist.csv", delimiter=',')

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

    window_size = 30
    step_size = 1

    all_fcs = []
    for i in range(0, ts.shape[0] - window_size, step_size):
        window_ts = ts[i:i + window_size, :]

        fc = np.corrcoef(window_ts, rowvar=False)
        fc[fc == 1] = 0.9999999
        all_fcs.append(fc)

    n_wins = len(all_fcs)
    FCD = np.zeros((n_wins, n_wins))
    for i in range(n_wins):
        for j in range(n_wins):
            FCD[i, j] = FCuCorrelation(all_fcs[i], all_fcs[j], fisher=True)
    hist = np.histogram(FCD, 100)[0]

    return kstest(hist, np.array(hc_hists).mean(axis=0)).statistic