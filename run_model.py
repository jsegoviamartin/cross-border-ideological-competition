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

# Deterministic version.
# Run the script with arguments from the command line.


import argparse
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
        #rate at which agents cease to be potential voters of country 1 because death or migration (e.g. 0.06)
        self.mu2 = mu2
        #rate at which agents enter voting system of country 2(e.g. 0.06, around 6% of the population is 18 yo.)
        self.mu3 = mu3
        #rate at which agents cease to be potential voters of country 2 because death or migration (e.g. 0.06)
        self.mu4 = mu4
        #rate at which agents cease to vote party B because death or migration (e.g. 0.06)
        self.muB = muB
        #rate at which agents cease to vote party C because death or migration (e.g. 0.06)
        self.muC = muC
        #rate at which agents cease to vote party D because death or migration (e.g. 0.06)
        self.muD = muD
        #rate at which agents cease to vote party E because death or migration (e.g. 0.06)
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

def main(args):
    # Initial conditions
    V10 = args.V10
    B0 = args.B0
    C0 = args.C0
    V20 = args.V20
    D0 = args.D0
    E0 = args.E0

    # Passing parameters to the model
    model = VBC(mu1=args.mu1, mu2=args.mu2, mu3=args.mu3, mu4=args.mu4,
                muB=args.muB, muC=args.muC, muD=args.muD, muE=args.muE,
                k1=args.k1, k2=args.k2, k3=args.k3, k4=args.k4,
                p1=args.p1, p2=args.p2, p3=args.p3, p4=args.p4,
                gamma1=args.gamma1, gamma2=args.gamma2, gamma3=args.gamma3, gamma4=args.gamma4,
                phi1=args.phi1, phi2=args.phi2, phi3=args.phi3, phi4=args.phi4)

    solver = RungeKutta4(model)
    solver.set_initial_condition([V10, B0, C0, V20, D0, E0])
    time_points = np.linspace(0, 200, 1001)
    u, t = solver.solve(time_points)
    V1 = u[:, 0]
    B = u[:, 1]
    C = u[:, 2]
    V2 = u[:, 3]
    D = u[:, 4]
    E = u[:, 5]
    print("V1:", V1)
    print("B:", B)
    print("C:", C)
    print("V2:", V2)
    print("D:", D)
    print("E:", E)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VBC model simulation.")
    parser.add_argument("--V10", type=float, default=1000, help="Initial V1 value")
    parser.add_argument("--B0", type=float, default=1000, help="Initial B value")
    parser.add_argument("--C0", type=float, default=1000, help="Initial C value")
    parser.add_argument("--V20", type=float, default=1000, help="Initial V2 value")
    parser.add_argument("--D0", type=float, default=1000, help="Initial D value")
    parser.add_argument("--E0", type=float, default=1000, help="Initial E value")
    parser.add_argument("--mu1", type=float, default=0.016, help="mu1 parameter")
    parser.add_argument("--mu2", type=float, default=0.016, help="mu2 parameter")
    parser.add_argument("--mu3", type=float, default=0.016, help="mu3 parameter")
    parser.add_argument("--mu4", type=float, default=0.016, help="mu4 parameter")
    parser.add_argument("--muB", type=float, default=0.016, help="muB parameter")
    parser.add_argument("--muC", type=float, default=0.016, help="muC parameter")
    parser.add_argument("--muD", type=float, default=0.016, help="muD parameter")
    parser.add_argument("--muE", type=float, default=0.016, help="muE parameter")
    parser.add_argument("--k1", type=float, default=0.5, help="k1 parameter")
    parser.add_argument("--k2", type=float, default=0.5, help="k2 parameter")
    parser.add_argument("--k3", type=float, default=0.5, help="k3 parameter")
    parser.add_argument("--k4", type=float, default=0.5, help="k4 parameter")
    parser.add_argument("--p1", type=float, default=0.1, help="p1 parameter")
    parser.add_argument("--p2", type=float, default=0.1, help="p2 parameter")
    parser.add_argument("--p3", type=float, default=0.1, help="p3 parameter")
    parser.add_argument("--p4", type=float, default=0.1, help="p4 parameter")
    parser.add_argument("--gamma1", type=float, default=0.01, help="gamma1 parameter")
    parser.add_argument("--gamma2", type=float, default=0.01, help="gamma2 parameter")
    parser.add_argument("--gamma3", type=float, default=0.01, help="gamma3 parameter")
    parser.add_argument("--gamma4", type=float, default=0.01, help="gamma4 parameter")
    parser.add_argument("--phi1", type=float, default=0.02, help="phi1 parameter")
    parser.add_argument("--phi2", type=float, default=0.02, help="phi2 parameter")
    parser.add_argument("--phi3", type=float, default=0.02, help="phi3 parameter")
    parser.add_argument("--phi4", type=float, default=0.02, help="phi4 parameter")
    args = parser.parse_args()
    main(args)
