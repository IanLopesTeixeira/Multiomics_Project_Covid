from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

def def_p(t):
    x=df.confirmados[tp_vector[1]]/df.obitos[tp_vector[1]+14]*.0066
    for i in range(len(tp_vector)-2):
        a=df.confirmados[tp_vector[i+1]]/df.obitos[tp_vector[i+1]+14]*.0066
        b=df.confirmados[tp_vector[i+2]]/df.obitos[tp_vector[i+2]+14]*.0066
        x=x+(b-a)*np.heaviside(t-tp_vector[i+2],1)
    return x

def PG_t(t_at, t_vector, pg_vector):
    x=pg_vector[0]*np.heaviside(t_at-t_vector[0],1)
    for i in range(len(t_vector)-1):
        x=x+(pg_vector[i+1]-pg_vector[i])*np.heaviside(t_at-t_vector[i+1],1)
    return x

def Pt_CVD19_ODE(t,y, R_0, a, e, g, a1, a2, v, r, t_vector, pg_vector, option):
    #Variables
    
    pv, S, E, Ic, Im, R = y    
 
    b0 = R_0*g/(p+(1-p)*a)
    pg= PG_t(t, t_vector, pg_vector)
    #print(pg,pv)
    # print(pg)
    # print(pv)
    # b_t = b0 * (1-pg)*(1-pv)
    a1=.99
    if option==0: 
        b_t = b0 * (1-pg)
    elif option==1: 
        b_t = b0 * (1-(pg*a1+pv*(1-a1)))
    else:
        b_t = b0 * ((1 - pg) * (1 - pv) + a1 * (1 - pg) * pv + a2 * (1 - pv) * pg)
    l = b_t * (S * Ic) + a* b_t * (S * Im)
    
    #Classes
    dpvt = - pv * (1 - pv) * v * (r - Ic*g)
    dSdt = - l
    dEdt = l - e*E
    dIcdt = p * e * E - g * Ic 
    dImdt = (1 - p) * e * E - g * Im
    dRdt = g * Ic + g * Im
    
    values = [dpvt, dSdt, dEdt, dIcdt, dImdt, dRdt]
    
    return values

def fun(x, y, R_0, ti, tf, pv0, a, e, g, a1, a2, v, r, t_vector, pg_vector, option, index, E0opt, pv0opt, pvopt, pgopt): #x=(E0, v, r, pg_vector)
    
    #inital conditions for solver
    p=def_p(tf)
    pv0if, E0if, a1if, a2if, vif, rif, pg_vectorif = x
    if E0opt==1:
        E0 = E0if
        Ic0 = p*E0
        Im0 = (1-p)*E0                              #Infectious (unconfirmed, Mild symptoms) 
        R0  = 0                                     #Removed
        S0  = 1 - Ic0 - Im0 - E0 - R0
    else : 
        [S0, E0, Ic0, Im0, R0]=y[1:]
    if pv0opt==1:
        pv0=pv0if
    if pvopt==1: 
        v = vif
        r = rif 
        a1 = a1if
        a2 = a2if
    if pgopt==1: pg_vector[index]=pg_vectorif
    #print(x)
    
    #parameters for solver
          
    day = np.array(range(ti, tf + 1))
    
    soln = solve_ivp(Pt_CVD19_ODE, [ti, tf], [pv0, S0, E0, Ic0, Im0, R0],
                 args = (R_0, a, e, g, a1, a2, v, r, t_vector, pg_vector,option),
                 t_eval = day)

    Ic = soln.y[3]
    
    return Ic*g*10**7-df.confirmados_novos[ti:tf+1]

#real data
df=pd.read_csv('data.csv', sep=',',header=0)

#simulation time
t_vector   = np.array([15,22,37,67,110,188,257,288,310,324,332])
tp_vector  = np.array([0,100,200,310])
tpv_vector = np.concatenate(([t_vector[0]], np.array([int((t_vector[i+1]+t_vector[i])*.5) for i in range(len(t_vector)-1)]),[t_vector[-1]]))

ti = t_vector[0]         #ti=11 corresponde ao dia 8 de março de 2020, em que há pela primeira vez, 9 casos novos confirmados
tf = t_vector[1]#t_vector[-1]#int((t_vector[0]+t_vector[1])/2)         #tf=21 é o último dia antes do primeiro estado de emergência
day = np.array(range(ti, tf + 1))

#definição do vetor de tempo das mudanças de confinamento

# index=[(ma.tg_vector[i]>ti and ma.tg_vector[i]<tf) for i in range(len(ma.tg_vector))]
# t_vector = np.array([ti])
# pg_vector = np.array([])
# j=0
# for i in range(len(ma.tg_vector)):
#     if index[i]==True:
#         j=i
#         t_vector=np.append(t_vector,ma.tg_vector[i])
#         pg_vector=np.append(pg_vector,ma.pg_vector[i-1])
# pg_vector=np.append(pg_vector,ma.pg_vector[j])


pg_vector = np.full(len(t_vector),.5)


#biologic parameters
R_0 = 2.5     #Basic number reproduction
e   = 1/3.8     #inverse of     E       --> Ic,Im rate
g   = 1/3.4     #inverse of     Ic,Im   --> R rate

a   = 1         #Ic/Im force of infecion ratio

#proportion of  E       --> Ic
p=def_p(t_vector[1])


#confinement parameters
v   = 10000      #speed of opinion change in voluntary confinement
r   = 100/10**7#opinion change threshold in voluntary confinement
a1  = 0.9        #reduction of infection due to voluntary confinement eficacy
a2  = 0.00001         #reduction of infection due to state confinement eficacy

#Initial Values
E0  = df.confirmados_novos[t_vector[0]]/(p*g*10**7)  #Exposed
#E0 = 1.1137432030524923e-05
Ic0 = p*E0                                  #Infectious (Confirmed, with symptoms)
Im0 = (1-p)*E0                              #Infectious (unconfirmed, Mild symptoms) 
R0  = 0                                     #Removed
S0  = 1 - Ic0 - Im0 - E0 - R0               #Susceptible
pv0 = 10000/10**7                             #Proportion of population willing to voluntary confinement

y= [pv0,E0,S0,Ic0,Im0,R0]

#index=0
#vetor de parametros a otimizar
Opt_0=np.array([pv0, E0, a1, a2, v, r, pg_vector[0]])


#variação admissível dos parâmetros a otimizar
Opt_m=np.array([0, 0, 0, 0, 0, 0, 0])


Opt_M=np.array([1, 1, 1, 1, 10**6, 1, 1])


#otimização pelos mínimos quadrados e definição do valor do confinamento pelo governo
res = least_squares(fun, Opt_0, bounds=(Opt_m, Opt_M),  
                    xtol=None, 
                    ftol=None,
                    method='dogbox',
                    args=(y, R_0, t_vector[0], t_vector[1], pv0, a, e, g, a1, a2, v, r, t_vector, pg_vector,2, 0,1,0,1,1)
                    )
#solver depois da otimização
pvo0, Eo0, a1o, a2o, vo,ro, pg_vector[0] = res.x
Ico0 = p*Eo0                                 #Infectious (Confirmed, with symptoms)
Imo0 = (1-p)*Eo0                              #Infectious (unconfirmed, Mild symptoms) 
So0  = 1 - Ico0 - Imo0 - Eo0 - R0 

solno = solve_ivp(Pt_CVD19_ODE, [t_vector[0], t_vector[1]], [pvo0, So0, Eo0, Ico0, Imo0, R0],
                  args = (R_0, a, e, g, a1o, a2o, vo, ro, t_vector, pg_vector, 2),
                  t_eval = day)


t = solno.t
pv = solno.y[0]
S = solno.y[1]
E = solno.y[2]
Ic = solno.y[3]
Im = solno.y[4]
R = solno.y[5]


for i in range(len(t_vector)-2):
    
    pv0i = solno.y[0][-1]
    S0i = solno.y[1][-1]
    E0i = solno.y[2][-1]
    Ic0i = solno.y[3][-1]
    Im0i = solno.y[4][-1]
    R0i = solno.y[5][-1]
    v=res.x[2]
    r=res.x[3]
    
    Opt_0=np.array([pv0, S0, a1, a2, v, r, pg_vector[i]])
    y=[pv0i,S0i,E0i,Ic0i,Im0i,R0i]
    
    res = least_squares(fun, Opt_0, bounds=(Opt_m, Opt_M),  
                        xtol=None, 
                        ftol=None,
                        method='dogbox',
                        args=(y, R_0, t_vector[i+1], t_vector[i+2], pv0, a, e, g, a1, a2, v, r, t_vector, pg_vector,2, i+1,0,0,1,1)
                        )
    
    pvi0, Ei0, a1i, a2i, vi, ri, pg_vector[i+1] = res.x
    day = np.array(range(t_vector[i+1], t_vector[i+2] + 1))
    
    solno = solve_ivp(Pt_CVD19_ODE, [t_vector[i+1], t_vector[i+2]], y,
                      args = (R_0, a, e, g, a1i, a2i, vi, ri, t_vector, pg_vector,2),
                      t_eval = day)
  
    t=np.concatenate((t,np.delete(solno.t,[0])))
    pv=np.concatenate((pv,np.delete(solno.y[0],[0])))
    S=np.concatenate((S,np.delete(solno.y[1],[0])))
    E=np.concatenate((E,np.delete(solno.y[2],[0])))
    Ic=np.concatenate((Ic,np.delete(solno.y[3],[0])))
    Im=np.concatenate((Im,np.delete(solno.y[4],[0])))
    R=np.concatenate((R,np.delete(solno.y[5],[0])))


Opt_0v=np.array([pv0, Eo0, a1, a2, v, r, pg_vector[0]])

res = least_squares(fun, Opt_0v, bounds=(Opt_m, Opt_M),  
                    xtol=None, 
                    ftol=None,
                    method='dogbox',
                    args=(y, R_0, tpv_vector[0], tpv_vector[1], pv0, a, e, g, a1, a2, v, r, t_vector, pg_vector,1, 0,0,1,1,0)
                    )
#solver depois da otimização
pvv0, Ev0, a1v, a2v, vv,rv, pg_vector_fake = res.x
day = np.array(range(tpv_vector[0], tpv_vector[1] + 1))


solno = solve_ivp(Pt_CVD19_ODE, [tpv_vector[0], tpv_vector[1]], [pvv0, So0, Eo0, Ico0, Imo0, R0],
                  args = (R_0, a, e, g, a1v, a2v, vv, rv, t_vector, pg_vector, 1),
                  t_eval = day)
tv = solno.t
pvv = solno.y[0]
Sv = solno.y[1]
Ev = solno.y[2]
Icv = solno.y[3]
Imv = solno.y[4]
Rv = solno.y[5]


for i in range(len(tpv_vector)-2):
    
    pv0vi = solno.y[0][-1]
    S0vi = solno.y[1][-1]
    E0vi = solno.y[2][-1]
    Ic0vi = solno.y[3][-1]
    Im0vi = solno.y[4][-1]
    R0vi = solno.y[5][-1]
    vv=res.x[2]
    rv=res.x[3]
    
    Opt_0=np.array([pv0vi, S0, a1v, a2v, vv, rv, pg_vector[i]])
    y=[pv0vi,S0vi,E0vi,Ic0vi,Im0vi,R0vi]
    
    res = least_squares(fun, Opt_0, bounds=(Opt_m, Opt_M),  
                        xtol=None, 
                        ftol=None,
                        method='dogbox',
                        args=(y, R_0, tpv_vector[i+1], tpv_vector[i+2], pv0, a, e, g, a1, a2, v, r, t_vector, pg_vector,1, i+1,0,0,1,0)
                        )
    
    pvv0, Ev0, a1v, a2v, vv,rv, pg_vector_fake = res.x
    day = np.array(range(tpv_vector[i+1], tpv_vector[i+2] + 1))
    
    solno = solve_ivp(Pt_CVD19_ODE, [tpv_vector[i+1], tpv_vector[i+2]], y,
                      args = (R_0, a, e, g, a1v, a2v, vv, rv, t_vector, pg_vector,1),
                      t_eval = day)
  
    tv=np.concatenate((tv,np.delete(solno.t,[0])))
    pvv=np.concatenate((pvv,np.delete(solno.y[0],[0])))
    Sv=np.concatenate((Sv,np.delete(solno.y[1],[0])))
    Ev=np.concatenate((Ev,np.delete(solno.y[2],[0])))
    Icv=np.concatenate((Icv,np.delete(solno.y[3],[0])))
    Imv=np.concatenate((Imv,np.delete(solno.y[4],[0])))
    Rv=np.concatenate((Rv,np.delete(solno.y[5],[0])))




#Plot
plt.rc("axes", labelsize=15, labelweight="bold", titlesize=20, titleweight="bold")
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
fig = plt.figure(facecolor="w")


    
    
ax = fig.add_subplot(111)
ax.plot(t, pv, 'k', alpha=0.5, lw=2, label='pv')
ax.plot(t, PG_t(t, t_vector, pg_vector), 'grey', alpha=0.5, lw=2, label='pg')
ax.set_title('Pg and Pv Evolution')

ax.set_xlabel('Time /days')
ax.set_ylabel('Percentage of population')
#ax.plot(t, S, 'g', alpha=0.5, lw=2, label='Susceptible')
#ax.plot(t, E, 'y', alpha=0.5, lw=2, label='Exposed')
legend = ax.legend()

plt.rc("axes", labelsize=15, labelweight="bold", titlesize=20, titleweight="bold")
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
fig = plt.figure(facecolor="w")
ax = fig.add_subplot(111)
ax.plot(t, g*Ic, 'r', alpha=0.5, lw=2, label='Newly Infected Confirmed after optimization')
ax.plot(t, df.confirmados_novos[t_vector[0]:t_vector[-1]+1]/10**7,'yellow', alpha=0.5, lw=2, label='Newly Infected Confirmed Real')
#ax.plot(t, (df.confirmados_novos[29:tf+1]/10**7).to_numpy()+res.fun/10**7, 'g', alpha=0.5, lw=2, label='Infected Confirmed after opt with residue')
ax.set_xlabel('Time /days')
ax.set_ylabel('Proportion population')
ax.set_title('Model 1 Behavior') 
#ax.set_ylim(0,1.2)
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
    
 