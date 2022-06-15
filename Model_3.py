# -*- coding: utf-8 -*-
"""
Created on Sat May 28 16:39:41 2022

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


# Total population, N.
N = 1*10**7

# Contact rate, beta, incubation rate, epsilon, and mean recovery rate tau.
 
R_0, sigma, gamma, alpha = 2.5, 1/3.8, 1/3.4, 1


# Times of change on transmission choose 

breaks = [7, 26, 96, 125, 201, 231, 256, 300, 308]

#  Reduction on beta
coeff = [0, 0.70164667, 0.6575286, 0.56353521, 0.30644124, 0.42921465, 0.60469562, 0.46198547]
# Velocity of changing opinion
velocities = [0]

# Threshold of changing opinion
thresholds = [0]

# First moment with no optimization

#break_0, coeff_0, velocity_0, threshold_0 = 7, 0, 0, 0


# Pg branch analysis
#groups = [0, 2, 4, 7]
groups = [0, 1, 3, 5, 8]

coeff_final = [0]

##### FUNCTIONS #################

def calculate_rho(t, s = 60):
        if t < s:
            t = max([20, s])
        return (total_cases[t] - total_cases[t-s])/(total_deaths[t+14] - total_deaths[t+14 - s]) * 0.0066
    
# The SEIR model differantial equations
def seir_f(t, y, N, R_0, sigma, gamma, alpha, rho, breaks, coeff, velocity, threshold):  
    
    pv, S, E, Ic, Iu, R= y
    
    rho = calculate_rho(int(t))
    
    beta_0 = (R_0*gamma)/(rho+(1-rho)*alpha)
    
    pg = np.piecewise(t, [t >= b for b in breaks], [c for c in coeff])
    
    c = pg + pv
    beta = beta_0 * (1 - c)
    
    l = beta * S  * (Ic + alpha * Iu)

    dpvt = - (pv + pg) * (1- pg - pv) * velocity * (threshold - Ic * gamma)
    dSdt = - l
    dEdt = l - sigma * E
    dIcdt= rho * sigma * E - gamma * Ic
    dIudt= (1-rho) * sigma * E - gamma * Iu
    dRdt = gamma * (Iu + Ic)

    return dpvt, dSdt, dEdt, dIcdt, dIudt, dRdt



def fun(x, y0, TI, TF, N, R_0, sigma, gamma, alpha, rho, breaks, total_cases):
    
    coeff = x[:-2]
    velocity = x[-2]
    threshold = x[-1]
    
    
    day = np.array(range(TI,TF+1))
    Ic_1 = np.array([1.0] * len(day))
    sol = solve_ivp(seir_f, [TI, TF], y0, args= (N, R_0, sigma, gamma, alpha,rho,breaks, coeff, velocity, threshold), t_eval=day)
    
    pv, S, E, Ic, Iu, R = sol.y
    
    Ic_1[0:len(Ic)] = Ic
                        
    return  gamma*Ic_1 - new_cases[TI:TF+1]/N 
 

##### SCRIPT ####################

rho = calculate_rho(30)

# Initial number of infected and recovered individuals, E0, Ic0, Im0 and R0.
E0 = 1400/N
Ic0, Iu0, R0 =  rho * E0, (1-rho) * E0, 0
 
# Everyone else, s0, is susceptible to infection initially.
S0 = 1 - E0 - Ic0 - Iu0 - R0

#Voluntary Lockdown doesn't exist before government
pv0 = 0

y0 = pv0, S0, E0, Ic0, Iu0, R0

day = np.array(range(breaks[0], breaks[1]))
sol = solve_ivp(seir_f, [breaks[0],breaks[1]], y0, args= (N, R_0,sigma, gamma, alpha, rho, [breaks[0]], [coeff[0]], velocities[0], thresholds[0]), t_eval = day)  

tv  = sol.t
pvv  = sol.y[0]
Sv  = sol.y[1]
Ev  = sol.y[2]
Icv = sol.y[3]
Iuv = sol.y[4]
Rv  = sol.y[5]

S0 = Sv[-1]
E0 = Ev[-1]
Ic0 = Icv[-1]
Iu0 = Iuv[-1]
R0 = Rv[-1]
pv0 = pvv[-1]

velocity = 5000
threshold = 200/N

for ix in range(1, len(groups)-1):
    # Initial conditions vector
    y0 = pv0, S0, E0, Ic0, Iu0, R0
    
    #optimizing at coeff in day

    coeff_analyzis = coeff[groups[ix]:groups[ix+1]]
    TI = breaks[groups[ix]]
    break_analyzis = []
    break_analyzis.extend(breaks[groups[ix]:groups[ix+1]])
    
    TF_analyzis = breaks[groups[ix+1]]
    
    Opt_m = np.array([0] * len(coeff_analyzis))
    Opt_m = np.append(Opt_m, [0, 0])
    Opt_M = np.array([1 ] * len(coeff_analyzis))
    Opt_M = np.append(Opt_M, [N, 1])
    Opt_0 = np.array(coeff_analyzis)
    Opt_0 = np.append(Opt_0, [velocity, threshold])
    
    
    print(break_analyzis, coeff_analyzis)
    ebrar = least_squares(fun, Opt_0, 
                                 bounds=(Opt_m,Opt_M), args=(y0, TI, TF_analyzis, N, R_0, sigma, gamma, alpha, rho, break_analyzis, total_cases)) 

    coeff_final.extend(ebrar.x[:-2])
    
    velocities.append(ebrar.x[-2])
    
    thresholds.append(ebrar.x[-1])
    
    # Integrate the SEIR equations over the time grid, t
    
    day = np.array(range(TI,TF_analyzis+1))
    sol = solve_ivp(seir_f, [TI, TF_analyzis], y0, args= (N, R_0,sigma, gamma, alpha, rho, break_analyzis, ebrar.x[:-2], velocities[-1], thresholds[-1]), t_eval = day)  
    
    tv  = np.concatenate((tv, np.delete(sol.t, [0])))
    pvv  = np.concatenate((pvv, np.delete(sol.y[0], [0])))
    Sv  = np.concatenate((Sv, np.delete(sol.y[1], [0])))
    Ev  = np.concatenate((Ev, np.delete(sol.y[2], [0])))
    Icv = np.concatenate((Icv, np.delete(sol.y[3], [0])))
    Iuv = np.concatenate((Iuv, np.delete(sol.y[4], [0])))
    Rv  = np.concatenate((Rv, np.delete(sol.y[5], [0])))
    
    S0 = Sv[-1]
    E0 = Ev[-1]
    Ic0 = Icv[-1]
    Iu0 = Iuv[-1]
    R0 = Rv[-1]
    pv0 = pvv[-1]
    
    velocity = velocities[-1]
    hreshold = thresholds[-1]
    
    
####    
plt.rc("axes", labelsize=15, labelweight="bold", titlesize=20, titleweight="bold")
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

fig = plt.figure(facecolor="w") 
ax = fig.add_subplot(111, axisbelow=True)
ax.plot(tv, gamma*Icv*N, 'r', lw=2, label='Newly Infected Confirmed after Optimization')
ax.plot(data['confirmados_novos'].iloc[breaks[0]:(tv[-1])+1], color = 'yellow',  lw=2, label='Newly Infected Confirmed Real')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number (1000s)')
ax.set_title('Model 3 Behavior')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()

fig = plt.figure(facecolor="w")
ax = fig.add_subplot(111, axisbelow=True)
ax.plot(tv, pvv, '', lw=2, label='Pv')
ax.plot(tv, np.piecewise(tv/1, [tv >= b for b in breaks[:-1]], [c for c in coeff_final]), '', lw=2, label='Pg')
ax.plot(tv, np.piecewise(tv/1, [tv >= b for b in breaks[:-1]], [c for c in coeff_final]) + pvv, '', lw=2, label='C')
ax.set_xlabel('Time /days')
ax.set_ylabel('Percentage')
ax.set_title('Pg and Pv Evolution')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()

fig = plt.figure(facecolor="w")
ax = fig.add_subplot(111, axisbelow=True)
ax.plot(tv, np.piecewise(tv/1, [ tv >= breaks[g] for g in groups[:-1]], [v for v in velocities]), '', lw=2, label='Velocity')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number (1000s)')
ax.set_title('Velocity Evolution')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()

fig = plt.figure(facecolor="w")
ax = fig.add_subplot(111, axisbelow=True)
ax.plot(tv, np.piecewise(tv/1, [tv >= breaks[g] for g in groups[:-1]], [t for t in thresholds]), '', lw=2, label='Threshold')
ax.set_xlabel('Time /days')
ax.set_ylabel('Percentage')
ax.set_title('Threshold Evolution')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()