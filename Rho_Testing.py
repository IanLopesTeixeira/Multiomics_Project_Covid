# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 16:03:04 2022

@author: Acer
"""
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

###### DATA & VARIABLES ##########

# real data
data = pd.read_csv("data.csv", sep=",", header=0)

total_cases = data["confirmados"].iloc[:]
new_cases = data["confirmados_novos"].iloc[:]
total_deaths = data["obitos"].iloc[:]
total_recovered = data["recuperados"].iloc[:]

fig = plt.figure(facecolor="w") #manter s = 60/70
plt.rc("axes", labelsize=15, labelweight="bold", titlesize=20, titleweight="bold")
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
ax = fig.add_subplot(axisbelow=True)
for s in (20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120):
    tv = np.array(range(0, 457))
    
    def rho(t, s):
        if t < s:
            t = max([20, s])
        return (total_cases[t] - total_cases[t-s])/(total_deaths[t+14] - total_deaths[t+14 - s]) * 0.0066
    
    calculate_rho = np.vectorize(rho)
    
    ax.plot(tv, calculate_rho(tv, s), '', lw=2, label=f'{s}')
    
ax.set_title('Rho Evolution')

ax.set_xlabel('Time /days')
ax.set_ylabel('Percentage')
#ax.plot(t, S, 'g', alpha=0.5, lw=2, label='Susceptible')
#ax.plot(t, E, 'y', alpha=0.5, lw=2, label='Exposed')
legend = ax.legend()


def calculate_rho(t):
    x = total_cases[tp_vector[1]]/total_deaths[tp_vector[1] + 14] * .0066
    for ix in range(len(tp_vector)-2):
        b = (total_cases[tp_vector[ix + 2]] - total_cases[tp_vector[ix + 1]])/(total_deaths[tp_vector[ix + 2]+14] - total_deaths[tp_vector[ix + 1]+14]) * 0.0066
        a = (total_cases[tp_vector[ix + 1]] - total_cases[tp_vector[ix]])/(total_deaths[tp_vector[ix + 1]+14] - total_deaths[tp_vector[ix]+14]) * 0.0066
        x += (b - a) * np.heaviside(t-tp_vector[ix + 1], 1)
    return x
pf = []
tv = []
tp_vector = np.array([0, 100, 200, 332])
t_vector = np.array([14, 21, 68, 96, 202, 232, 257, 302])

for ix in range(t_vector[0], t_vector[-1]+1):
    pf.append(calculate_rho(ix))
    tv.append(ix)

# PLOTS
fig = plt.figure(figsize=(5,10))

ax = fig.add_subplot(axisbelow=True)
ax.set_title('Rho Evolution')

ax.set_xlabel('Time /days')
ax.set_ylabel('Percentage')
legend = ax.legend()
ax.plot(tv, pf, 'k', alpha=0.5, lw=2, label='Rho')
ax.set_xlabel('Time /days')
ax.set_ylabel('Percentage')
legend = ax.legend()