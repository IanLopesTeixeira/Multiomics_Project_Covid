# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:22:11 2022

@author: ian
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


# Total population, N.
N = 1*10**7

# Contact rate, beta, incubation rate, epsilon, and mean recovery rate tau.
 
R_0, sigma, gamma, alpha = 2.5, 1/3.8, 1/3.4, 1.2


# Times of change on transmission choose 

breaks = [26, 96, 125, 201, 231, 256, 300]

#  Reduction on beta
coeff = [0.8, 0.8, 0.25, 0.35, 0.8, 0.2, 0.2]

# First moment with no optimization

break_0, coeff_0 = 7, 0

#Rho timepoints
tp_vector = np.array([0, 100, 200, 332])

# Pg branch analysis
groups = [1, 3]
 
# Final time for the simulation
TF = 308


##### FUNCTIONS #################
def calculate_rho(t):
    x = total_cases[tp_vector[1]]/total_deaths[tp_vector[1] + 14] * .0066
    for ix in range(len(tp_vector)-2):
        b = (total_cases[tp_vector[ix + 2]] - total_deaths[tp_vector[ix + 1]])/(total_deaths[tp_vector[ix + 2]+14] - total_deaths[tp_vector[ix + 1]+14]) * 0.0066
        a = (total_cases[tp_vector[ix + 1]] - total_deaths[tp_vector[ix]])/(total_deaths[tp_vector[ix + 1]+14] - total_deaths[tp_vector[ix]+14]) * 0.0066
        x += (b - a) * np.heaviside(t-tp_vector[ix + 1], 1)
    return x

rho = calculate_rho(tp_vector[0])

# Initial number of infected and recovered individuals, E0, Ic0, Im0 and R0.
E0 = 1400/N
Ic0, Iu0, R0 =  rho * E0, (1-rho) * E0, 0
 
# Everyone else, s0, is susceptible to infection initially.
S0 = 1 - E0 - Ic0 - Iu0 - R0
 

# The SEIR model differantial equations
def seir_f(t, y, N, R_0, sigma, gamma, alpha, rho, breaks, coeff):  
    
    S, E, Ic, Iu, R= y
    
    rho = calculate_rho(t)
    
    beta_0 = (R_0*gamma)/(rho+(1-rho)*alpha)
    
    beta = np.piecewise(t, [t >= b for b in breaks], [(1 - c) * beta_0 for c in coeff])
    
    l = beta * S  * (Ic + alpha * Iu)
    
    dSdt = - l
    dEdt = l - sigma * E
    dIcdt= rho * sigma * E - gamma * Ic
    dIudt= (1-rho) * sigma * E - gamma * Iu
    dRdt = gamma * (Iu + Ic)
 
    return dSdt, dEdt, dIcdt, dIudt, dRdt

def fun(x, y0, TI, TF, N, R_0, sigma, gamma, alpha, rho, breaks, total_cases, coeff0):
    
    
    coeff = [coeff0]
    coeff = np.append(coeff, x)
    
    
    day = np.array(range(TI,TF+1))
    sol = solve_ivp(seir_f, [TI, TF], y0, args= (N, R_0, sigma, gamma, alpha,rho,breaks, coeff), t_eval=day)
    
    S, E, Ic, Iu, R = sol.y
    
    return  gamma*Ic - new_cases[TI:TF+1]/N 
 

##### SCRIPT ####################

rho = calculate_rho(tp_vector[0])

# Initial number of infected and recovered individuals, E0, Ic0, Im0 and R0.
E0 = 1400/N
Ic0, Iu0, R0 =  rho * E0, (1-rho) * E0, 0
 
# Everyone else, s0, is susceptible to infection initially.
S0 = 1 - E0 - Ic0 - Iu0 - R0
 
# Initial conditions vector
y0 = S0, E0, Ic0, Iu0, R0


#optimizing at coeff in day 

coeff_analyzis = coeff[0:groups[0]+1]

TI = break_0

TF_analyzis = breaks[groups[0] + 1] - 1

break_analyzis = [break_0]
break_analyzis.extend(breaks[:groups[0]+1])


ebrar = least_squares(fun, coeff_analyzis, 
                             bounds=(0,1), args=(y0, TI, TF_analyzis, N, R_0, sigma, gamma, alpha, rho, break_analyzis, total_cases, coeff_0)) 
coeff_analyzis = [coeff_0]
coeff_analyzis = np.append(coeff_analyzis, ebrar.x)
coeff_final = coeff_analyzis


# Integrate the SEIR equations over the time grid, t

day = np.array(range(TI,TF_analyzis+1))
sol = solve_ivp(seir_f, [TI, TF_analyzis], y0, args= (N, R_0,sigma, gamma, alpha,rho,break_analyzis, coeff_analyzis), t_eval = day)  

Sv, Ev, Icv, Iuv, Rv = sol.y                
tv = sol.t


for ix in range(1, len(groups)):
    Sv0 = Sv[-1]
    Ev0 = Ev[-1]
    Icv0 = Icv[-1]
    Iuv0 = Iuv[-1]
    Rv0 = Rv[-1]
    
    coeff_analyzis = coeff[groups[ix-1]+1:groups[ix]+1]
    
    TI = breaks[groups[ix-1]+1] - 1
    TF_analyzis = breaks[groups[ix] + 1] - 1
    
    break_analyzis = []
    break_analyzis.extend(breaks[groups[ix-1]:groups[ix]+1])
    y0 = Sv0, Ev0, Icv0, Iuv0, Rv0
    #optimizing at coeff in day 
    
    ebrar = least_squares(fun, coeff_analyzis, 
                                 bounds=(0,1), args=(y0, TI, TF_analyzis, N, R_0, sigma, gamma, alpha, rho, break_analyzis, total_cases, coeff_final[-1])) 

    coeff_analyzis = [coeff_final[-1]]
    coeff_analyzis = np.append(coeff_analyzis, ebrar.x)
    coeff_final = np.concatenate((coeff_final, np.delete(coeff_analyzis, [0])))

    
    # Integrate the SEIR equations over the time grid, t
    
    day = np.array(range(TI,TF_analyzis+1))
    sol = solve_ivp(seir_f, [TI, TF_analyzis], y0, args= (N, R_0,sigma, gamma, alpha,rho,break_analyzis, coeff_analyzis), t_eval= day)  
    

    tv  = np.concatenate((tv, np.delete(sol.t,    [0])))
    Sv  = np.concatenate((Sv, np.delete(sol.y[0], [0])))
    Ev  = np.concatenate((Ev, np.delete(sol.y[1], [0])))
    Icv = np.concatenate((Icv, np.delete(sol.y[2], [0])))
    Iuv = np.concatenate((Iuv, np.delete(sol.y[3], [0])))
    Rv  = np.concatenate((Rv, np.delete(sol.y[4], [0])))

    
Sv0 = Sv[-1]
Ev0 = Ev[-1]
Icv0 = Icv[-1]
Iuv0 = Iuv[-1]
Rv0 = Rv[-1]


coeff_analyzis = coeff[groups[-1]+1:]

TI = breaks[groups[-1]+1] - 1
TF_analyzis = TF
 
break_analyzis = breaks[groups[-1]:]
y0 = Sv0, Ev0, Icv0, Iuv0, Rv0
#optimizing at coeff in day 

ebrar = least_squares(fun, coeff_analyzis, 
                             bounds=(0,1), args=(y0, TI, TF_analyzis, N, R_0, sigma, gamma, alpha, rho, break_analyzis, total_cases, coeff_final[-1])) 

coeff_analyzis = [coeff_final[-1]]
coeff_analyzis = np.append(coeff_analyzis, ebrar.x)
coeff_final = np.concatenate((coeff_final, np.delete(coeff_analyzis, [0])))


# Integrate the SEIR equations over the time grid, t
day = np.array(range(TI,TF_analyzis+1))
sol = solve_ivp(seir_f, [TI, TF_analyzis], y0, args= (N, R_0,sigma, gamma, alpha,rho,break_analyzis, coeff_analyzis), t_eval = day)  

tv  = np.concatenate((tv, np.delete(sol.t,    [0])))
Sv  = np.concatenate((Sv, np.delete(sol.y[0], [0])))
Ev  = np.concatenate((Ev, np.delete(sol.y[1], [0])))
Icv = np.concatenate((Icv, np.delete(sol.y[2], [0])))
Iuv = np.concatenate((Iuv, np.delete(sol.y[3], [0])))
Rv  = np.concatenate((Rv, np.delete(sol.y[4], [0])))


#### PLOTS ########
breaks_f = [break_0]
breaks_f.extend(breaks)
fs = np.array([float(x) for x in tv])
R_tv= np.piecewise(fs, [fs >= b for b in breaks_f], [(1- c)*R_0 for c in coeff_final])

# Plot the data on separate curves for s(t), e(t), i(t) and r(t)
plt.rc("axes", labelsize=15, labelweight="bold", titlesize=20, titleweight="bold")
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
fig = plt.figure(facecolor="w") 
ax = fig.add_subplot(111,axisbelow=True)
ax.plot(tv, gamma*Icv*N, 'r', lw=2, label='Newly Infected Confirmed after Optimization')
ax.plot(data['confirmados_novos'].iloc[break_0:TF+1], 'yellow', lw=2, label='Newly Infected Confirmed Real')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number (1000s)')
ax.set_title("Model 2 Behavior")
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
legend = ax.legend()

fig = plt.figure(facecolor="w") 
ax = fig.add_subplot(111, axisbelow=True)
ax.plot(tv,  np.piecewise(tv/1, [tv >= b for b in breaks_f], [c for c in coeff_final]), color='grey', lw = 2, label = "Pg")
ax.set_xlabel('Time /days')
ax.set_ylabel('Percentage of Population')
ax.set_title('Pg Evolution')
legend = ax.legend()

   

fig = plt.figure(facecolor="w") 
ax = fig.add_subplot(111, axisbelow=True)
ax.plot(tv, R_tv, color='green', lw = 2, label = "Rt")
ax.set_xlabel('Time /days')
ax.set_ylabel('Rt value')
ax.set_title('Rt evolution')
legend = ax.legend()







