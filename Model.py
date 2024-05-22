# """
# Copyright 2024 Jose Segovia-Martin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# For any use of this software, proper citation must be given to the creator,
# Jose Segovia-Martin, acknowledging the original source.
# """

# Cross-border influence model: 2-country with 2-party system model of political competition.

# The model we present here idealises ideologies as fixed and as competing with each other for supporters both
# within and across borders.
# Agents can only support one ideology (party or political tendency) at any given moment in time.
# Stochastic differential process for each political affiliation and fit model solutions to population growth rates
# and voting populations in US presidential elections from 1932 to 2020.
# we assume the chance of an agent of coming  into contact with a party n of another
# country is given by the proportion of voters of party n in such other country. Stronger ideologies
# or parties within a country are also better able to export their ideas than minority parties.



from ODESolver import RungeKutta4
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


# Deterministic model (2 countries): Parameters and governing equations

class VBC:
    def __init__(self, mu1, mu2, mu3, mu4, muB, muC, muD, muE, k1, k2, k3, k4,
                 p1, p2, p3, p4, gamma1, gamma2, gamma3, gamma4, phi1, phi2, phi3, phi4):
        #rate at which agents enter voting system of country 1 (e.g. 0.06, around 6% of the population is 18 yo.)
        self.mu1 = mu1
        #rate at which agents cease to be potential voters of country 1 because deth or migration (e.g. 0.06)
        self.mu2 = mu2
        #rate at which agents enter voting system of country 2(e.g. 0.06, around 6% of the population is 18 yo.)
        self.mu3 = mu3
        #rate at which agents cease to be potential voters of country 2 because deth or migration (e.g. 0.06)
        self.mu4 = mu4
        ##rate at which agents cease to vote party B because death or migration (e.g. 0.06)
        self.muB = muB
        ##rate at which agents cease to vote party C because death or migration (e.g. 0.06)
        self.muC = muC
        ##rate at which agents cease to vote party D because death or migration (e.g. 0.06)
        self.muD = muD
        ##rate at which agents cease to vote party E because death or migration (e.g. 0.06)
        self.muE = muE
        #average number of contacts per time per capita of party B (e.g. party B reaches 10% of the population, 0.1)
        self.k1 = k1
        #average number of contacts per time per capita of party C (e.g. party C reaches 30% of the population, 0.3)
        self.k2 = k2
        #average number of contacts per time per capita of party D (e.g. party B reaches 10% of the population, 0.1)
        self.k3 = k3
        #average number of contacts per time per capita of party E (e.g. party C reaches 30% of the population, 0.3)
        self.k4 = k4
        #probability of B convincing another agent per contact (between 0.0-1.0)
        self.p1 = p1
        #probability of C convincing another agent per contact(between 0.0-1.0)
        self.p2 = p2
        #probability of D convincing another agent per contact (between 0.0-1.0)
        self.p3 = p3
        #probability of E convincing another agent per contact (between 0.0-1.0)
        self.p4 = p4
        #per capita leakage of agents from party B (between 0.0-1.0)
        self.gamma1 = gamma1
        #per capita leakage of agents from party C (between 0.0-1.0)
        self.gamma2 = gamma2
        # per capita leakage of agents from party D (between 0.0-1.0)
        self.gamma3 = gamma3
        # per capita leakage of agents from party E (between 0.0-1.0)
        self.gamma4 = gamma4
        #per capita recruitment of party B from party C (between 0.0-1.0)
        self.phi1 = phi1
        # per capita recruitment of party C from party D (between 0.0-1.0)
        self.phi2 = phi2
        # per capita recruitment of party D from party E (between 0.0-1.0)
        self.phi3 = phi3
        # per capita recruitment of party E from party D (between 0.0-1.0)
        self.phi4 = phi4

    def __call__(self,u,t):
        #Unknown function
        V1, B, C, V2, D, E = u
        # Country 1: V1 -> Potential voters, B -> Voters of Political Party B, C -> Voters of Political Party C
        # Original population size of country 1 at t0
        N1=V1+B+C
        # Country 2: V2 -> Potential voters, D -> Voters of Political Party D, E -> Voters of Political Party E
        ##Original population size of country 2 at t0
        N2 = V2 + D + E
        # Governing equations country 1
        dV1 = self.mu1*N1\
              - self.k1*self.p1*V1*(B/N1)\
              - (1-(self.k1*self.p1))*self.k3*self.p3*V1*(D/N2)\
              - self.k2*self.p2*V1*(C/N1)\
              - (1-(self.k2*self.p2))*self.k4*self.p4*V1*(E/N2)\
              - self.mu2*V1\
              + self.gamma1*B\
              + self.gamma2*C
        dB = self.k1*self.p1*V1*(B/N1)\
             + (1-(self.k1*self.p1))*self.k3*self.p3*V1*(D/N2)\
             - self.phi2*B*(C/N1)\
             - (1-self.phi2)*self.phi4*B*(E/N2)\
             + self.phi1*C*(B/N1)\
             + (1-self.phi1)*self.phi3*C*(D/N2)\
             - self.muB*B \
             - self.gamma1*B
        dC = self.k2*self.p2*V1*(C/N1) \
             + (1-(self.k2*self.p2))*self.k4*self.p4*V1*(E/N2)\
             - self.phi1*C*(B/N1) \
             - (1 - self.phi1)*self.phi3*C*(D/N2)\
             + self.phi2*B*(C/N1)\
             + (1-self.phi2)*self.phi4*B*(E/N2)\
             - self.muC*C\
             - self.gamma2*C
        #Governing equations country 2
        dV2 = self.mu3*N2\
              - self.k3*self.p3*V2*(D/N2) \
              - (1-(self.k3*self.p3))*self.k1*self.p1*V2*(B/N1)\
              - self.k4*self.p4*V2*(E/N2) \
              - (1-(self.k4*self.p4))*self.k2*self.p2*V2*(C/N1)\
              - self.mu4*V2\
              + self.gamma3*D\
              + self.gamma4*E
        dD = self.k3*self.p3*V2*(D/N2) \
             + (1-(self.k3*self.p3))*self.k1*self.p1*V2*(B/N1)\
             - self.phi4*D*(E/N2) \
             - (1-self.phi4)*self.phi2*D*(C/N1)\
             + self.phi3*E*(D/N2) \
             + (1-self.phi3)*self.phi1*E*(B/N1)\
             - self.muD*D\
             - self.gamma3*D
        dE = self.k4*self.p4*V2*(E/N2) \
             + (1-(self.k4*self.p4))*self.k2*self.p2*V2*(C/N1) \
             - self.phi3*E*(D/N2) \
             - (1-self.phi3)*self.phi1*E*(B/N1)\
             + self.phi4*D*(E/N2) \
             + (1-self.phi4)*self.phi2*D*(C/N1) \
             - self.muE*E\
             - self.gamma4*E
        return [dV1,dB,dC,dV2,dD,dE]

# First simulation
#Initial conditions country 1
V10 = 1000
B0 = 1000
C0 = 1000
#Initial conditions country 2
V20 = 1000
D0 = 1000
E0 = 1000
#Passing parameters to the model
model = VBC(mu1=0.016, mu2=0.016, mu3=0.016, mu4=0.016,
            muB= 0.016, muC= 0.016, muD=0.016, muE=0.016,
            k1= 0.5, k2= 0.5, k3=0.5, k4=0.5,
            p1= 0.1, p2= 0.1, p3=0.1, p4=0.1,
            gamma1= 0.01, gamma2= 0.01,gamma3=0.01, gamma4=0.01,
            phi1= 0.02, phi2= 0.02, phi3=0.02, phi4=0.02)

solver= RungeKutta4(model)
solver.set_initial_condition([V10,B0,C0,V20,D0,E0])
time_points = np.linspace(0, 200, 1001)
u,t = solver.solve(time_points)
V1 = u[:,0]; B = u[:,1]; C = u[:,2]; V2 = u[:,3]; D = u[:,4]; E = u[:,5]
print(V1,B,C,V2,D,E)

fig, axs = plt.subplots(3,2)
fig.suptitle('')
axs[0,0].plot(t,V1,label="V1")
axs[0,0].plot(t,B,label="B", ls=(0, (5, 1)))
axs[0,0].plot(t,C,label="C", ls="-.")
axs[0,0].plot(t,V2,label="V2", ls="--")
axs[0,0].plot(t,D,label="D", ls="--")
axs[0,0].plot(t,E,label="E", ls=":")
axs[0,0].set_xlabel('Time in years')
axs[0,0].set_ylabel('Number of agents')
axs[0,0].set_title("")
#axs[0,0].legend(loc='upper right')

# Second simulation
#Initial conditions country 1
V10 = 1000
B0 = 1000
C0 = 1000
#Initial conditions country 2
V20 = 500
D0 = 500
E0 = 500
#Passing parameters to the model
model = VBC(mu1=0.016, mu2=0.016, mu3=0.016, mu4=0.016,
            muB= 0.016, muC= 0.016, muD=0.016, muE=0.016,
            k1= 0.5, k2= 0.5, k3=0.5, k4=0.5,
            p1= 0.1, p2= 0.1, p3=0.1, p4=0.1,
            gamma1= 0.01, gamma2= 0.01,gamma3=0.01, gamma4=0.01,
            phi1= 0.02, phi2= 0.02, phi3=0.02, phi4=0.02)


solver= RungeKutta4(model)
solver.set_initial_condition([V10,B0,C0,V20,D0,E0])
time_points = np.linspace(0, 200, 1001)
u,t = solver.solve(time_points)
V1 = u[:,0]; B = u[:,1]; C = u[:,2]; V2 = u[:,3]; D = u[:,4]; E = u[:,5]

axs[0,1].plot(t,V1,label="V1")
axs[0,1].plot(t,B,label="B", ls=(0, (5, 1)))
axs[0,1].plot(t,C,label="C", ls="-.")
axs[0,1].plot(t,V2,label="V2", ls="--")
axs[0,1].plot(t,D,label="D", ls="--")
axs[0,1].plot(t,E,label="E", ls=":")
axs[0,1].set_xlabel('Time in years')
axs[0,1].set_ylabel('Number of agents')

# Third simulation
#Initial conditions country 1
V10 = 1000
B0 = 1000
C0 = 1000
#Initial conditions country 2
V20 = 1000
D0 = 1000
E0 = 1000
#Passing parameters to the model
model = VBC(mu1=0.016, mu2=0.016, mu3=0.016, mu4=0.016,
            muB= 0.016, muC= 0.016, muD=0.016, muE=0.016,
            k1= 0.4, k2= 0.5, k3=0.5, k4=0.5,
            p1= 0.1, p2= 0.1, p3=0.1, p4=0.1,
            gamma1= 0.01, gamma2= 0.01,gamma3=0.01, gamma4=0.01,
            phi1= 0.02, phi2= 0.02, phi3=0.02, phi4=0.02)

solver= RungeKutta4(model)
solver.set_initial_condition([V10,B0,C0,V20,D0,E0])
time_points = np.linspace(0, 200, 1001)
u,t = solver.solve(time_points)
V1 = u[:,0]; B = u[:,1]; C = u[:,2]; V2 = u[:,3]; D = u[:,4]; E = u[:,5]
print(V1,B,C,V2,D,E)

axs[1,0].plot(t,V1,label="V1")
axs[1,0].plot(t,B,label="B", ls=(0, (5, 1)))
axs[1,0].plot(t,C,label="C", ls="-.")
axs[1,0].plot(t,V2,label="V2", ls="--")
axs[1,0].plot(t,D,label="D", ls="--")
axs[1,0].plot(t,E,label="E", ls=":")
axs[1,0].set_xlabel('Time in years')
axs[1,0].set_ylabel('Number of agents')
axs[1,0].set_title("")
#axs[0,0].legend(loc='upper right')

# Fourth simulation
#Initial conditions country 1
V10 = 1000
B0 = 1000
C0 = 1000
#Initial conditions country 2
V20 = 500
D0 = 500
E0 = 500
#Passing parameters to the model
model = VBC(mu1=0.016, mu2=0.016, mu3=0.016, mu4=0.016,
            muB= 0.016, muC= 0.016, muD=0.016, muE=0.016,
            k1= 0.4, k2= 0.5, k3=0.5, k4=0.5,
            p1= 0.1, p2= 0.1, p3=0.1, p4=0.1,
            gamma1= 0.01, gamma2= 0.01,gamma3=0.01, gamma4=0.01,
            phi1= 0.02, phi2= 0.02, phi3=0.02, phi4=0.02)


solver= RungeKutta4(model)
solver.set_initial_condition([V10,B0,C0,V20,D0,E0])
time_points = np.linspace(0, 200, 1001)
u,t = solver.solve(time_points)
V1 = u[:,0]; B = u[:,1]; C = u[:,2]; V2 = u[:,3]; D = u[:,4]; E = u[:,5]

axs[1,1].plot(t,V1,label="V1")
axs[1,1].plot(t,B,label="B", ls=(0, (5, 1)))
axs[1,1].plot(t,C,label="C", ls="-.")
axs[1,1].plot(t,V2,label="V2", ls="--")
axs[1,1].plot(t,D,label="D", ls="--")
axs[1,1].plot(t,E,label="E", ls=":")
axs[1,1].set_xlabel('Time in years')
axs[1,1].set_ylabel('Number of agents')


# Fifth simulation
#Initial conditions country 1
V10 = 10000
B0 = 10000
C0 = 10000
#Initial conditions country 2
V20 = 5000
D0 = 5000
E0 = 5000
model = VBC(mu1=0.016, mu2=0.016, mu3=0.016, mu4=0.016,
            muB= 0.016, muC= 0.016, muD=0.016, muE=0.016,
            k1= 0.6, k2= 0.4, k3=0.6, k4=0.6,
            p1= 0.2, p2= 0.1, p3=0.1, p4=0.2,
            gamma1= 0.01, gamma2= 0.01,gamma3=0.01, gamma4=0.01,
            phi1= 0.01, phi2= 0.03, phi3=0.015, phi4=0.01)

solver= RungeKutta4(model)
solver.set_initial_condition([V10,B0,C0,V20,D0,E0])
time_points = np.linspace(0, 200, 1001)
u,t = solver.solve(time_points)
V1 = u[:,0]; B = u[:,1]; C = u[:,2]; V2 = u[:,3]; D = u[:,4]; E = u[:,5]
print(V1,B,C,V2,D,E)

axs[2,0].plot(t,V1,label="V1")
axs[2,0].plot(t,B,label="B", ls=(0, (5, 1)))
axs[2,0].plot(t,C,label="C", ls="-.")
axs[2,0].plot(t,V2,label="V2", ls="--")
axs[2,0].plot(t,D,label="D", ls="--")
axs[2,0].plot(t,E,label="E", ls=":")
axs[2,0].set_xlabel('Time in years')
axs[2,0].set_ylabel('Number of agents')


# Sixth simulation
#Initial conditions country 1
V10 = 10000
B0 = 10000
C0 = 10000
#Initial conditions country 2
V20 = 5000
D0 = 5000
E0 = 5000
model = VBC(mu1=0.016, mu2=0.016, mu3=0.016, mu4=0.016,
            muB= 0.016, muC= 0.016, muD=0.016, muE=0.016,
            k1= 0.6, k2= 0.4, k3=0.6, k4=0.6,
            p1= 0.2, p2= 0.1, p3=0.1, p4=0.2,
            gamma1= 0.01, gamma2= 0.01,gamma3=0.01, gamma4=0.01,
            phi1= 0.01, phi2= 0.03, phi3=0.03, phi4=0.01)


solver= RungeKutta4(model)
solver.set_initial_condition([V10,B0,C0,V20,D0,E0])
time_points = np.linspace(0, 200, 1001)
u,t = solver.solve(time_points)
V1 = u[:,0]; B = u[:,1]; C = u[:,2]; V2 = u[:,3]; D = u[:,4]; E = u[:,5]

axs[2,1].plot(t,V1,label="V1")
axs[2,1].plot(t,B,label="B", ls=(0, (5, 1)))
axs[2,1].plot(t,C,label="C", ls="-.")
axs[2,1].plot(t,V2,label="V2", ls="--")
axs[2,1].plot(t,D,label="D", ls="--")
axs[2,1].plot(t,E,label="E", ls=":")
axs[2,1].set_xlabel('Time in years')
axs[2,1].set_ylabel('Number of agents')

handles, labels = axs[1,1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2)
plt.show()
fig.set_size_inches(7, 6)
fig.subplots_adjust(bottom=0.2)
fig.savefig('3dPlot.pdf')

