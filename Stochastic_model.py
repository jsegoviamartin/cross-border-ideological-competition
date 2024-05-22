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
import random
from collections import OrderedDict
import pandas as pd

# Load data from CSV
file_path = 'Voting_data.csv'
data = pd.read_csv(file_path)


class VBC:
    def __init__(self, r1, r2, mu1, mu2, mu3, mu4, muB, muC, muD, muE,
                 k1, k2, k3, k4, p1, p2, p3, p4, gamma1, gamma2, gamma3, gamma4, phi1, phi2, phi3, phi4,
                 mu1t, mu2t, mu3t, mu4t, muBt, muCt, muDt, muEt,
                 k1t, k2t, k3t, k4t, p1t, p2t, p3t, p4t, gamma1t, gamma2t, gamma3t, gamma4t, phi1t, phi2t, phi3t, phi4t
                 ):
        # population rate at which mu changes over time in country 1 (if mu1=mu2=muB=muC,then r1 is the population growth rate)
        self.r1 = r1
        # population rate at which mu changes over time in country 2 (if mu3=mu4=muD=muE,then r1 is the population growth rate)
        self.r2 = r2
        # rate at which agents enter voting system of country 1 (e.g. 0.06, around 6% of the population is 18 yo.)
        self.mu1 = mu1
        # rate at which agents cease to be potential voters of country 1 because deth or migration (e.g. 0.06)
        self.mu2 = mu2
        # rate at which agents enter voting system of country 2(e.g. 0.06, around 6% of the population is 18 yo.)
        self.mu3 = mu3
        # rate at which agents cease to be potential voters of country 2 because deth or migration (e.g. 0.06)
        self.mu4 = mu4
        ##rate at which agents cease to vote party B because death or migration (e.g. 0.06)
        self.muB = muB
        ##rate at which agents cease to vote party C because death or migration (e.g. 0.06)
        self.muC = muC
        ##rate at which agents cease to vote party D because death or migration (e.g. 0.06)
        self.muD = muD
        ##rate at which agents cease to vote party E because death or migration (e.g. 0.06)
        self.muE = muE
        # From the specified time cut-off: rate at which agents enter voting system of country 1 (e.g. 0.06, around 6% of the population is 18 yo.)
        self.mu1t = mu1t
        # From the specified time cut-off: rate at which agents cease to be potential voters of country 1 because deth or migration (e.g. 0.06)
        self.mu2t = mu2t
        # From the specified time cut-off: rate at which agents enter voting system of country 2(e.g. 0.06, around 6% of the population is 18 yo.)
        self.mu3t = mu3t
        # From the specified time cut-off: rate at which agents cease to be potential voters of country 2 because deth or migration (e.g. 0.06)
        self.mu4t = mu4t
        # From the specified time cut-off: rate at which agents cease to vote party B because death or migration (e.g. 0.06)
        self.muBt = muBt
        # From the specified time cut-off: rate at which agents cease to vote party C because death or migration (e.g. 0.06)
        self.muCt = muCt
        # From the specified time cut-off: rate at which agents cease to vote party D because death or migration (e.g. 0.06)
        self.muDt = muDt
        # From the specified time cut-off: rate at which agents cease to vote party E because death or migration (e.g. 0.06)
        self.muEt = muEt
        # average number of contacts per time per capita of party B (e.g. party B reaches 10% of the population, 0.1)
        self.k1 = k1
        # average number of contacts per time per capita of party C (e.g. party C reaches 30% of the population, 0.3)
        self.k2 = k2
        # average number of contacts per time per capita of party D (e.g. party B reaches 10% of the population, 0.1)
        self.k3 = k3
        # average number of contacts per time per capita of party E (e.g. party C reaches 30% of the population, 0.3)
        self.k4 = k4
        # probability of B convincing another agent per contact (between 0.0-1.0)
        self.p1 = p1
        # probability of C convincing another agent per contact(between 0.0-1.0)
        self.p2 = p2
        # probability of D convincing another agent per contact (between 0.0-1.0)
        self.p3 = p3
        # probability of E convincing another agent per contact (between 0.0-1.0)
        self.p4 = p4
        # per capita leakage of agents from party B (between 0.0-1.0)
        self.gamma1 = gamma1
        # per capita leakage of agents from party C (between 0.0-1.0)
        self.gamma2 = gamma2
        # per capita leakage of agents from party D (between 0.0-1.0)
        self.gamma3 = gamma3
        # per capita leakage of agents from party E (between 0.0-1.0)
        self.gamma4 = gamma4
        # per capita recruitment of party B from party C (between 0.0-1.0)
        self.phi1 = phi1
        # per capita recruitment of party C from party D (between 0.0-1.0)
        self.phi2 = phi2
        # per capita recruitment of party D from party E (between 0.0-1.0)
        self.phi3 = phi3
        # per capita recruitment of party E from party D (between 0.0-1.0)
        self.phi4 = phi4
        # From t=88 (that is, 2020) on...
        # average number of contacts per time per capita of party B (e.g. party B reaches 10% of the population, 0.1)
        self.k1t = k1t
        # average number of contacts per time per capita of party C (e.g. party C reaches 30% of the population, 0.3)
        self.k2t = k2t
        # average number of contacts per time per capita of party D (e.g. party B reaches 10% of the population, 0.1)
        self.k3t = k3t
        # average number of contacts per time per capita of party E (e.g. party C reaches 30% of the population, 0.3)
        self.k4t = k4t
        # probability of B convincing another agent per contact (between 0.0-1.0)
        self.p1t = p1t
        # probability of C convincing another agent per contact(between 0.0-1.0)
        self.p2t = p2t
        # probability of D convincing another agent per contact (between 0.0-1.0)
        self.p3t = p3t
        # probability of E convincing another agent per contact (between 0.0-1.0)
        self.p4t = p4t
        # per capita leakage of agents from party B (between 0.0-1.0)
        self.gamma1t = gamma1t
        # per capita leakage of agents from party C (between 0.0-1.0)
        self.gamma2t = gamma2t
        # per capita leakage of agents from party D (between 0.0-1.0)
        self.gamma3t = gamma3t
        # per capita leakage of agents from party E (between 0.0-1.0)
        self.gamma4t = gamma4t
        # per capita recruitment of party B from party C (between 0.0-1.0)
        self.phi1t = phi1t
        # per capita recruitment of party C from party D (between 0.0-1.0)
        self.phi2t = phi2t
        # per capita recruitment of party D from party E (between 0.0-1.0)
        self.phi3t = phi3t
        # per capita recruitment of party E from party D (between 0.0-1.0)
        self.phi4t = phi4t

    def __call__(self, u, t):
        # Unknown function
        V1, B, C, V2, D, E = u
        # Country 1: V1 -> Potential voters, B -> Voters of Political Party B, C -> Voters of Political Party C
        # Original population size of country 1 at t0
        N1 = V1 + B + C
        # Country 2: V2 -> Potential voters, D -> Voters of Political Party D, E -> Voters of Political Party E
        ##Original population size of country 2 at t0
        N2 = V2 + D + E
        if isinstance(t, float):
            # population growth rate over time: f(pop growth)=bo+b1*t.
            # Population growth rate in country 1 (US)
            self.r1 = 0.018 - 0.0001 * t
            # Population growth rate in country 2 (Rest of the world)
            self.r2 = 0.022 - 0.0001 * t
            # print(self.r)

        # "while" includes governing equations from t0 to t88, i.e. fitted to US data from 1932 to 2020
        while np.any(t < 88):
            # Governing equations country 1
            dV1 = (self.mu1+self.r1) * N1 \
                  - self.k1 * self.p1 * V1 * (B / N1) + self.k1 * self.p1 * V1 * (B / N1) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                  - (1 - (self.k1 * self.p1)) * self.k3 * self.p3 * V1 * (D / N2) + (
                          1 - (self.k1 * self.p1)) * self.k3 * self.p3 * V1 * (D / N2) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                  - self.k2 * self.p2 * V1 * (C / N1) + self.k2 * self.p2 * V1 * (C / N1) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                  - (1 - (self.k2 * self.p2)) * self.k4 * self.p4 * V1 * (E / N2) + (
                          1 - (self.k2 * self.p2)) * self.k4 * self.p4 * V1 * (E / N2) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                  - self.mu2 * V1 \
                  + self.gamma1 * B + self.gamma1 * B * sqrtdt * np.random.normal(0.0, 1.0) \
                  + self.gamma2 * C + self.gamma2 * C * sqrtdt * np.random.normal(0.0, 1.0)
            dB = self.k1 * self.p1 * V1 * (B / N1) + self.k1 * self.p1 * V1 * (B / N1) * sqrtdt * np.random.normal(0.0,
                                                                                                                   1.0) \
                 + (1 - (self.k1 * self.p1)) * self.k3 * self.p3 * V1 * (D / N2) + (
                         1 - (self.k1 * self.p1)) * self.k3 * self.p3 * V1 * (D / N2) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                 - self.phi2 * B * (C / N1) + self.phi2 * B * (C / N1) * sqrtdt * np.random.normal(0.0, 1.0) \
                 - (1 - self.phi2) * self.phi4 * B * (E / N2) + (1 - self.phi2) * self.phi4 * B * (
                         E / N2) * sqrtdt * np.random.normal(0.0, 1.0) \
                 + self.phi1 * C * (B / N1) + self.phi1 * C * (B / N1) * sqrtdt * np.random.normal(0.0, 1.0) \
                 + (1 - self.phi1) * self.phi3 * C * (D / N2) + (1 - self.phi1) * self.phi3 * C * (
                         D / N2) * sqrtdt * np.random.normal(0.0, 1.0) \
                 - self.muB * B \
                 - self.gamma1 * B + self.gamma1 * B * sqrtdt * np.random.normal(0.0, 1.0)
            dC = self.k2 * self.p2 * V1 * (C / N1) + self.k2 * self.p2 * V1 * (C / N1) * sqrtdt * np.random.normal(0.0,
                                                                                                                   1.0) \
                 + (1 - (self.k2 * self.p2)) * self.k4 * self.p4 * V1 * (E / N2) + (
                         1 - (self.k2 * self.p2)) * self.k4 * self.p4 * V1 * (E / N2) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                 - self.phi1 * C * (B / N1) + self.phi1 * C * (B / N1) * sqrtdt * np.random.normal(0.0, 1.0) \
                 - (1 - self.phi1) * self.phi3 * C * (D / N2) + (1 - self.phi1) * self.phi3 * C * (
                         D / N2) * sqrtdt * np.random.normal(0.0, 1.0) \
                 + self.phi2 * B * (C / N1) + self.phi2 * B * (C / N1) * sqrtdt * np.random.normal(0.0, 1.0) \
                 + (1 - self.phi2) * self.phi4 * B * (E / N2) + (1 - self.phi2) * self.phi4 * B * (
                         E / N2) * sqrtdt * np.random.normal(0.0, 1.0) \
                 - self.muC * C \
                 - self.gamma2 * C + self.gamma2 * C * sqrtdt * np.random.normal(0.0, 1.0)
            # Governing equations country 2
            dV2 = (self.mu3+self.r2) * N2 \
                  - self.k3 * self.p3 * V2 * (D / N2) + self.k3 * self.p3 * V2 * (D / N2) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                  - (1 - (self.k3 * self.p3)) * self.k1 * self.p1 * V2 * (B / N1) + (
                          1 - (self.k3 * self.p3)) * self.k1 * self.p1 * V2 * (B / N1) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                  - self.k4 * self.p4 * V2 * (E / N2) + self.k4 * self.p4 * V2 * (E / N2) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                  - (1 - (self.k4 * self.p4)) * self.k2 * self.p2 * V2 * (C / N1) + (
                          1 - (self.k4 * self.p4)) * self.k2 * self.p2 * V2 * (C / N1) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                  - self.mu4 * V2 \
                  + self.gamma3 * D + self.gamma3 * D * sqrtdt * np.random.normal(0.0, 1.0) \
                  + self.gamma4 * E + self.gamma4 * E * sqrtdt * np.random.normal(0.0, 1.0)
            dD = self.k3 * self.p3 * V2 * (D / N2) + self.k3 * self.p3 * V2 * (D / N2) * sqrtdt * np.random.normal(0.0,
                                                                                                                   1.0) \
                 + (1 - (self.k3 * self.p3)) * self.k1 * self.p1 * V2 * (B / N1) + (
                         1 - (self.k3 * self.p3)) * self.k1 * self.p1 * V2 * (B / N1) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                 - self.phi4 * D * (E / N2) + self.phi4 * D * (E / N2) * sqrtdt * np.random.normal(0.0, 1.0) \
                 - (1 - self.phi4) * self.phi2 * D * (C / N1) + (1 - self.phi4) * self.phi2 * D * (
                         C / N1) * sqrtdt * np.random.normal(0.0, 1.0) \
                 + self.phi3 * E * (D / N2) + self.phi3 * E * (D / N2) * sqrtdt * np.random.normal(0.0, 1.0) \
                 + (1 - self.phi3) * self.phi1 * E * (B / N1) + (1 - self.phi3) * self.phi1 * E * (
                         B / N1) * sqrtdt * np.random.normal(0.0, 1.0) \
                 - self.muD * D \
                 - self.gamma3 * D + self.gamma3 * D * sqrtdt * np.random.normal(0.0, 1.0)
            dE = self.k4 * self.p4 * V2 * (E / N2) + self.k4 * self.p4 * V2 * (E / N2) * sqrtdt * np.random.normal(0.0,
                                                                                                                   1.0) \
                 + (1 - (self.k4 * self.p4)) * self.k2 * self.p2 * V2 * (C / N1) + (
                         1 - (self.k4 * self.p4)) * self.k2 * self.p2 * V2 * (C / N1) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                 - self.phi3 * E * (D / N2) + self.phi3 * E * (D / N2) * sqrtdt * np.random.normal(0.0, 1.0) \
                 - (1 - self.phi3) * self.phi1 * E * (B / N1) + (1 - self.phi3) * self.phi1 * E * (
                         B / N1) * sqrtdt * np.random.normal(0.0, 1.0) \
                 + self.phi4 * D * (E / N2) + self.phi4 * D * (E / N2) * sqrtdt * np.random.normal(0.0, 1.0) \
                 + (1 - self.phi4) * self.phi2 * D * (C / N1) + (1 - self.phi4) * self.phi2 * D * (
                         C / N1) * sqrtdt * np.random.normal(0.0, 1.0) \
                 - self.muE * E \
                 - self.gamma4 * E + self.gamma4 * E * sqrtdt * np.random.normal(0.0, 1.0)
            return [dV1, dB, dC, dV2, dD, dE]

        # "else" includes governing equations from t89 onwards to t88, i.e. predictions for US from 2020 on.
        else:
            # Governing equations country 1
            dV1 = (self.mu1t+self.r1) * N1 \
                  - self.k1t * self.p1t * V1 * (B / N1) + self.k1t * self.p1t * V1 * (
                              B / N1) * sqrtdt * np.random.normal(0.0, 1.0) \
                  - (1 - (self.k1t * self.p1t)) * self.k3t * self.p3t * V1 * (D / N2) + (
                          1 - (self.k1t * self.p1t)) * self.k3t * self.p3t * V1 * (D / N2) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                  - self.k2t * self.p2t * V1 * (C / N1) + self.k2t * self.p2t * V1 * (
                              C / N1) * sqrtdt * np.random.normal(0.0, 1.0) \
                  - (1 - (self.k2t * self.p2t)) * self.k4t * self.p4t * V1 * (E / N2) + (
                          1 - (self.k2t * self.p2t)) * self.k4t * self.p4t * V1 * (E / N2) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                  - self.mu2t * V1 \
                  + self.gamma1t * B + self.gamma1t * B * sqrtdt * np.random.normal(0.0, 1.0) \
                  + self.gamma2t * C + self.gamma2t * C * sqrtdt * np.random.normal(0.0, 1.0)
            dB = self.k1t * self.p1t * V1 * (B / N1) + self.k1t * self.p1t * V1 * (B / N1) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                 + (1 - (self.k1t * self.p1t)) * self.k3t * self.p3t * V1 * (D / N2) + (
                         1 - (self.k1t * self.p1t)) * self.k3t * self.p3t * V1 * (D / N2) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                 - self.phi2t * B * (C / N1) + self.phi2t * B * (C / N1) * sqrtdt * np.random.normal(0.0, 1.0) \
                 - (1 - self.phi2t) * self.phi4t * B * (E / N2) + (1 - self.phi2t) * self.phi4t * B * (
                         E / N2) * sqrtdt * np.random.normal(0.0, 1.0) \
                 + self.phi1t * C * (B / N1) + self.phi1t * C * (B / N1) * sqrtdt * np.random.normal(0.0, 1.0) \
                 + (1 - self.phi1t) * self.phi3t * C * (D / N2) + (1 - self.phi1t) * self.phi3t * C * (
                         D / N2) * sqrtdt * np.random.normal(0.0, 1.0) \
                 - self.muBt * B \
                 - self.gamma1t * B + self.gamma1t * B * sqrtdt * np.random.normal(0.0, 1.0)
            dC = self.k2t * self.p2t * V1 * (C / N1) + self.k2t * self.p2t * V1 * (C / N1) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                 + (1 - (self.k2t * self.p2t)) * self.k4t * self.p4t * V1 * (E / N2) + (
                         1 - (self.k2t * self.p2t)) * self.k4t * self.p4t * V1 * (E / N2) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                 - self.phi1t * C * (B / N1) + self.phi1t * C * (B / N1) * sqrtdt * np.random.normal(0.0, 1.0) \
                 - (1 - self.phi1t) * self.phi3t * C * (D / N2) + (1 - self.phi1t) * self.phi3t * C * (
                         D / N2) * sqrtdt * np.random.normal(0.0, 1.0) \
                 + self.phi2t * B * (C / N1) + self.phi2t * B * (C / N1) * sqrtdt * np.random.normal(0.0, 1.0) \
                 + (1 - self.phi2t) * self.phi4t * B * (E / N2) + (1 - self.phi2t) * self.phi4t * B * (
                         E / N2) * sqrtdt * np.random.normal(0.0, 1.0) \
                 - self.muCt * C \
                 - self.gamma2t * C + self.gamma2t * C * sqrtdt * np.random.normal(0.0, 1.0)
            # Governing equations country 2
            dV2 = (self.mu3t+self.r2) * N2 \
                  - self.k3t * self.p3t * V2 * (D / N2) + self.k3t * self.p3t * V2 * (
                              D / N2) * sqrtdt * np.random.normal(0.0, 1.0) \
                  - (1 - (self.k3t * self.p3t)) * self.k1t * self.p1t * V2 * (B / N1) + (
                          1 - (self.k3t * self.p3t)) * self.k1t * self.p1t * V2 * (B / N1) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                  - self.k4t * self.p4t * V2 * (E / N2) + self.k4t * self.p4t * V2 * (
                              E / N2) * sqrtdt * np.random.normal(0.0, 1.0) \
                  - (1 - (self.k4t * self.p4t)) * self.k2t * self.p2t * V2 * (C / N1) + (
                          1 - (self.k4t * self.p4t)) * self.k2t * self.p2t * V2 * (C / N1) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                  - self.mu4t * V2 \
                  + self.gamma3t * D + self.gamma3t * D * sqrtdt * np.random.normal(0.0, 1.0) \
                  + self.gamma4t * E + self.gamma4t * E * sqrtdt * np.random.normal(0.0, 1.0)
            dD = self.k3t * self.p3t * V2 * (D / N2) + self.k3t * self.p3t * V2 * (D / N2) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                 + (1 - (self.k3t * self.p3t)) * self.k1t * self.p1t * V2 * (B / N1) + (
                         1 - (self.k3t * self.p3t)) * self.k1t * self.p1t * V2 * (B / N1) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                 - self.phi4t * D * (E / N2) + self.phi4t * D * (E / N2) * sqrtdt * np.random.normal(0.0, 1.0) \
                 - (1 - self.phi4t) * self.phi2t * D * (C / N1) + (1 - self.phi4t) * self.phi2t * D * (
                         C / N1) * sqrtdt * np.random.normal(0.0, 1.0) \
                 + self.phi3t * E * (D / N2) + self.phi3t * E * (D / N2) * sqrtdt * np.random.normal(0.0, 1.0) \
                 + (1 - self.phi3t) * self.phi1t * E * (B / N1) + (1 - self.phi3t) * self.phi1t * E * (
                         B / N1) * sqrtdt * np.random.normal(0.0, 1.0) \
                 - self.muDt * D \
                 - self.gamma3t * D + self.gamma3t * D * sqrtdt * np.random.normal(0.0, 1.0)
            dE = self.k4t * self.p4t * V2 * (E / N2) + self.k4t * self.p4t * V2 * (E / N2) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                 + (1 - (self.k4t * self.p4t)) * self.k2t * self.p2t * V2 * (C / N1) + (
                         1 - (self.k4t * self.p4t)) * self.k2t * self.p2t * V2 * (C / N1) * sqrtdt * np.random.normal(
                0.0, 1.0) \
                 - self.phi3t * E * (D / N2) + self.phi3t * E * (D / N2) * sqrtdt * np.random.normal(0.0, 1.0) \
                 - (1 - self.phi3t) * self.phi1t * E * (B / N1) + (1 - self.phi3t) * self.phi1t * E * (
                         B / N1) * sqrtdt * np.random.normal(0.0, 1.0) \
                 + self.phi4t * D * (E / N2) + self.phi4t * D * (E / N2) * sqrtdt * np.random.normal(0.0, 1.0) \
                 + (1 - self.phi4t) * self.phi2t * D * (C / N1) + (1 - self.phi4t) * self.phi2t * D * (
                         C / N1) * sqrtdt * np.random.normal(0.0, 1.0) \
                 - self.muEt * E \
                 - self.gamma4t * E + self.gamma4t * E * sqrtdt * np.random.normal(0.0, 1.0)
            return [dV1, dB, dC, dV2, dD, dE]

#Population
#US: Eligible voters, Democrats, Republicans
def multiples(value, length):
    return [value * i for i in range(1, length + 1)]
#Timeseries from 1932 to 2020
t_us = multiples(4,23)
#Population in millions
# Ensure that 'Non-partisan', 'Dem', and 'Rep' are the correct column names
Total_Abstention = data['Non-partisan'].tolist()
Total_US_dem = data['Dem'].tolist()
Total_US_rep = data['Rep'].tolist()

# First simulation
#Initial conditions country 1 (US)
V10 = 34650000
# B0 = 22821277
# C0 = 15761254
B0 = 22000000
C0 = 16000000
#Initial conditions country 2 # At t0 = 1932, for agents outside the U.S., it is assumed that
#political tendencies are evenly split at the start of the simulation, with one-third non-partisan, one-third
# pro-Democrats, and one-third pro-Republicans.
V20 = 50000000
D0 = 50000000
E0 = 50000000
model = VBC(r1=0.02,r2=0.02,
            mu1=0.017, mu2=0.017, mu3=0.017, mu4=0.017,
            muB= 0.017, muC= 0.017, muD=0.017, muE=0.017,
            k1= 0.55, k2= 0.55, k3=0.1, k4=0.1,
            p1= 0.15, p2= 0.15, p3=0.1, p4=0.1,
            gamma1= 0.01, gamma2= 0.01, gamma3=0.01, gamma4=0.01,
            phi1= 0.05, phi2= 0.05, phi3=0.01, phi4=0.01,
            mu1t=0.017, mu2t=0.017, mu3t=0.017, mu4t=0.017,
            muBt=0.017, muCt=0.017, muDt=0.017, muEt=0.017,
            k1t=0.55, k2t=0.55, k3t=0.1, k4t=0.1,
            p1t=0.15, p2t=0.15, p3t=0.1, p4t=0.1,
            gamma1t=0.01, gamma2t=0.01, gamma3t=0.01, gamma4t=0.01,
            phi1t=0.05, phi2t=0.05, phi3t=0.01, phi4t=0.01,
            )

fig, axs = plt.subplots(4, 3, figsize=(10,10))
#fig.suptitle('S0000')
#number of simulations
for i in range(10):
    solver = RungeKutta4(model)
    solver.set_initial_condition([V10, B0, C0, V20, D0, E0])
    dt = 0.2  # Time step
    T = 168  # Total time
    n = int(T / dt)  # Number of time steps
    time_points = np.linspace(0, T, n)  # Vector times
    sqrtdt = np.sqrt(dt)
    u, t = solver.solve(time_points)
    V1 = u[:, 0];
    B = u[:, 1];
    C = u[:, 2];
    V2 = u[:, 3];
    D = u[:, 4];
    E = u[:, 5]

    axs[0,0].title.set_text('S0000')
    axs[0, 0].plot(t, V1, label="Sim.Abs", color="yellow", alpha=0.1, zorder=1, linewidth=0.5)
    axs[0, 0].plot(t, B, label="Sim.Dem", color="blue", alpha=0.1, zorder=1, linewidth=0.5)
    axs[0, 0].plot(t, C, label="Sim.Rep", color="red", alpha=0.1, zorder=1, linewidth=0.5)
    axs[0, 0].scatter(t_us, np.array(Total_Abstention) * 1000000, label="Abs", color="yellow", s=4, zorder=2,
                      edgecolors= "black", linewidth=0.1)
    axs[0, 0].scatter(t_us, np.array(Total_US_dem) * 1000000, label="Dem", color="blue", s=4, zorder=2,
                      edgecolors= "black", linewidth=0.1)
    axs[0, 0].scatter(t_us, np.array(Total_US_rep) * 1000000, label="Rep", color="red", s=4, zorder=2,
                      edgecolors= "black", linewidth=0.1)
    #axs[0, 0].set_xlabel('Time in years')
    #axs[0, 0].set_ylabel('Number of agents')
    axs[0,0].set_ylim(0, 200000000)
    x = np.arange(0, 169, 42)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels([i + 1932 for i in x])
    axs[0, 0].set_xticklabels([])

# Second simulation
#Initial conditions country 1 (US)
V10 = 34650000
# B0 = 22821277
# C0 = 15761254
B0 = 22000000
C0 = 16000000
#C0 = 19000000
#Initial conditions country 2 # At t0 = 1932, for agents outside the U.S., it is assumed that
#political tendencies are evenly split at the start of the simulation, with one-third non-partisan, one-third
# pro-Democrats, and one-third pro-Republicans.
V20 = 50000000
D0 = 50000000
E0 = 50000000
model = VBC(r1=0.02,r2=0.02,
            mu1=0.017, mu2=0.017, mu3=0.017, mu4=0.017,
            muB= 0.017, muC= 0.017, muD=0.017, muE=0.017,
            k1= 0.55, k2= 0.55, k3=0.1, k4=0.1,
            p1= 0.15, p2= 0.15, p3=0.1, p4=0.1,
            gamma1= 0.01, gamma2= 0.01, gamma3=0.01, gamma4=0.01,
            phi1= 0.050, phi2= 0.055, phi3=0.01, phi4=0.01,
            mu1t=0.017, mu2t=0.017, mu3t=0.017, mu4t=0.017,
            muBt=0.017, muCt=0.017, muDt=0.017, muEt=0.017,
            k1t=0.55, k2t=0.55, k3t=0.1, k4t=0.1,
            p1t=0.15, p2t=0.15, p3t=0.1, p4t=0.1,
            gamma1t=0.01, gamma2t=0.01, gamma3t=0.01, gamma4t=0.01,
            phi1t=0.050, phi2t=0.055, phi3t=0.01, phi4t=0.01,
            )


for i in range(10):
    solver = RungeKutta4(model)
    solver.set_initial_condition([V10, B0, C0, V20, D0, E0])
    dt = 0.2  # Time step
    T = 168  # Total time
    n = int(T / dt)  # Number of time steps
    time_points = np.linspace(0, T, n)  # Vector times
    sqrtdt = np.sqrt(dt)
    u, t = solver.solve(time_points)
    V1 = u[:, 0];
    B = u[:, 1];
    C = u[:, 2];
    V2 = u[:, 3];
    D = u[:, 4];
    E = u[:, 5]

    axs[0, 1].title.set_text('S0100')
    axs[0, 1].plot(t, V1, label="Sim.Abs", color="yellow", alpha=0.1, zorder=1, linewidth=0.5)
    axs[0, 1].plot(t, B, label="Sim.Dem", color="blue", alpha=0.1, zorder=1, linewidth=0.5)
    axs[0, 1].plot(t, C, label="Sim.Rep", color="red", alpha=0.1, zorder=1, linewidth=0.5)
    axs[0, 1].scatter(t_us, np.array(Total_Abstention) * 1000000, label="Abs", color="yellow", s=4, zorder=2,
                      edgecolors= "black", linewidth=0.1)
    axs[0, 1].scatter(t_us, np.array(Total_US_dem) * 1000000, label="Dem", color="blue", s=4,  zorder=2, edgecolors=
    "black", linewidth=0.1)
    axs[0, 1].scatter(t_us, np.array(Total_US_rep) * 1000000, label="Rep", color="red", s=4,  zorder=2, edgecolors=
    "black", linewidth=0.1)
    #axs[0, 1].set_xlabel('Time in years')
    #axs[0, 1].set_ylabel('Number of agents')
    axs[0,1].set_yticklabels([])
    axs[0, 1].set_ylim(0, 200000000)
    x = np.arange(0, 169, 42)
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels([i + 1932 for i in x])
    axs[0, 1].set_xticklabels([])

# Third simulation
#Initial conditions country 1 (US)
V10 = 34650000
B0 = 19291265
C0 = 19291265
#C0 = 19000000
#Initial conditions country 2 # At t0 = 1932, for agents outside the U.S., it is assumed that
#political tendencies are evenly split at the start of the simulation, with one-third non-partisan, one-third
# pro-Democrats, and one-third pro-Republicans.
V20 = 50000000
D0 = 50000000
E0 = 50000000
model = VBC(r1=0.02,r2=0.02,
            mu1=0.017, mu2=0.017, mu3=0.017, mu4=0.017,
            muB= 0.017, muC= 0.017, muD=0.017, muE=0.017,
            k1= 0.55, k2= 0.55, k3=0.1, k4=0.1,
            p1= 0.15, p2= 0.15, p3=0.1, p4=0.1,
            gamma1= 0.01, gamma2= 0.01, gamma3=0.01, gamma4=0.01,
            phi1= 0.050, phi2= 0.050, phi3=0.01, phi4=0.01,
            mu1t=0.017, mu2t=0.017, mu3t=0.017, mu4t=0.017,
            muBt=0.017, muCt=0.017, muDt=0.017, muEt=0.017,
            k1t=0.55, k2t=0.55, k3t=0.1, k4t=0.1,
            p1t=0.15, p2t=0.15, p3t=0.1, p4t=0.1,
            gamma1t=0.01, gamma2t=0.01, gamma3t=0.01, gamma4t=0.01,
            phi1t=0.050, phi2t=0.050, phi3t=0.01, phi4t=0.01,
            )

for i in range(10):
    solver = RungeKutta4(model)
    solver.set_initial_condition([V10, B0, C0, V20, D0, E0])
    dt = 0.2  # Time step
    T = 168  # Total time
    n = int(T / dt)  # Number of time steps
    time_points = np.linspace(0, T, n)  # Vector times
    sqrtdt = np.sqrt(dt)
    u, t = solver.solve(time_points)
    V1 = u[:, 0];
    B = u[:, 1];
    C = u[:, 2];
    V2 = u[:, 3];
    D = u[:, 4];
    E = u[:, 5]

    axs[0, 2].title.set_text('S1000')
    axs[0, 2].plot(t, V1, label="Sim.Abs", color="yellow", alpha=0.1, zorder=1, linewidth=0.5)
    axs[0, 2].plot(t, B, label="Sim.Dem", color="blue", alpha=0.1, zorder=1, linewidth=0.5)
    axs[0, 2].plot(t, C, label="Sim.Rep", color="red", alpha=0.1, zorder=1, linewidth=0.5)
    # axs[0,2].plot(t,V2,label="V2", ls="--")
    # axs[0,2].plot(t,D,label="D", ls="--")
    # axs[0,2].plot(t,E,label="E", ls=":")
    axs[0, 2].scatter(t_us, np.array(Total_Abstention) * 1000000, label="Abs", color="yellow", s=4, zorder=2,
                      edgecolors= "black", linewidth=0.1)
    axs[0, 2].scatter(t_us, np.array(Total_US_dem) * 1000000, label="Dem", color="blue", s=4, zorder=2, edgecolors= "black", linewidth=0.1)
    axs[0, 2].scatter(t_us, np.array(Total_US_rep) * 1000000, label="Rep", color="red", s=4, zorder=2, edgecolors= "black", linewidth=0.1)
    # axs[0, 1].set_xlabel('Time in years')
    # axs[0, 1].set_ylabel('Number of agents')
    axs[0, 2].set_yticklabels([])
    axs[0, 2].set_ylim(0, 200000000)
    x = np.arange(0, 169, 42)
    axs[0, 2].set_xticks(x)
    axs[0, 2].set_xticklabels([i + 1932 for i in x])
    axs[0, 2].set_xticklabels([])


# Fourth simulation
#Initial conditions country 1 (US)
V10 = 34650000
# B0 = 22821277
# C0 = 15761254
B0 = 22000000
C0 = 16000000
#C0 = 19000000
#Initial conditions country 2 # At t0 = 1932, for agents outside the U.S., it is assumed that
#political tendencies are evenly split at the start of the simulation, with one-third non-partisan, one-third
# pro-Democrats, and one-third pro-Republicans.
V20 = 50000000
D0 = 50000000
E0 = 50000000
model = VBC(r1=0.02,r2=0.02,
            mu1=0.017, mu2=0.017, mu3=0.017, mu4=0.017,
            muB= 0.017, muC= 0.017, muD=0.017, muE=0.017,
            k1= 0.55, k2= 0.55, k3=0.1, k4=0.1,
            p1= 0.15, p2= 0.15, p3=0.1, p4=0.1,
            gamma1= 0.01, gamma2= 0.01, gamma3=0.01, gamma4=0.01,
            phi1= 0.050, phi2= 0.050, phi3=0.01, phi4=0.01,
            mu1t=0.017, mu2t=0.017, mu3t=0.017, mu4t=0.017,
            muBt=0.017, muCt=0.017, muDt=0.017, muEt=0.017,
            k1t=0.55, k2t=0.55, k3t=0.1, k4t=0.1,
            p1t=0.15, p2t=0.15, p3t=0.1, p4t=0.1,
            gamma1t=0.01, gamma2t=0.01, gamma3t=0.01, gamma4t=0.01,
            phi1t=0.050, phi2t=0.050, phi3t=0.015, phi4t=0.01,
            )


#number of simulations
for i in range(10):
    solver = RungeKutta4(model)
    solver.set_initial_condition([V10, B0, C0, V20, D0, E0])
    dt = 0.2  # Time step
    T = 168  # Total time
    n = int(T / dt)  # Number of time steps
    time_points = np.linspace(0, T, n)  # Vector times
    sqrtdt = np.sqrt(dt)
    u, t = solver.solve(time_points)
    V1 = u[:, 0];
    B = u[:, 1];
    C = u[:, 2];
    V2 = u[:, 3];
    D = u[:, 4];
    E = u[:, 5]

    axs[1, 0].title.set_text('S0001')
    axs[1, 0].plot(t, V1, label="Sim.Abs", color="yellow", alpha=0.1, zorder=1, linewidth=0.5)
    axs[1, 0].plot(t, B, label="Sim.Dem", color="blue", alpha=0.1, zorder=1, linewidth=0.5)
    axs[1, 0].plot(t, C, label="Sim.Rep", color="red", alpha=0.1, zorder=1, linewidth=0.5)
    axs[1, 0].scatter(t_us, np.array(Total_Abstention) * 1000000, label="Abs", color="yellow", s=4, zorder=2,
                      edgecolors= "black", linewidth=0.1)
    axs[1, 0].scatter(t_us, np.array(Total_US_dem) * 1000000, label="Dem", color="blue", s=4, zorder=2, edgecolors= "black", linewidth=0.1)
    axs[1, 0].scatter(t_us, np.array(Total_US_rep) * 1000000, label="Rep", color="red", s=4, zorder=2, edgecolors= "black", linewidth=0.1)
    #axs[1, 0].set_xlabel('Time in years')
    axs[1, 0].set_ylabel('   ')
    axs[1, 0].set_ylim(0, 200000000)
    x = np.arange(0, 169, 42)
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels([i + 1932 for i in x])
    axs[1, 0].patch.set_facecolor('blue')
    axs[1, 0].patch.set_alpha(0.05)
    axs[1, 0].set_xticklabels([])


# Fifth simulation
#Initial conditions country 1 (US)
V10 = 34650000
# B0 = 22821277
# C0 = 15761254
B0 = 22000000
C0 = 16000000
#C0 = 19000000
#Initial conditions country 2 # At t0 = 1932, for agents outside the U.S., it is assumed that
#political tendencies are evenly split at the start of the simulation, with one-third non-partisan, one-third
# pro-Democrats, and one-third pro-Republicans.
V20 = 50000000
D0 = 50000000
E0 = 50000000
model = VBC(r1=0.02,r2=0.02,
            mu1=0.017, mu2=0.017, mu3=0.017, mu4=0.017,
            muB= 0.017, muC= 0.017, muD=0.017, muE=0.017,
            k1= 0.55, k2= 0.55, k3=0.1, k4=0.1,
            p1= 0.15, p2= 0.15, p3=0.1, p4=0.1,
            gamma1= 0.01, gamma2= 0.01, gamma3=0.01, gamma4=0.01,
            phi1= 0.050, phi2= 0.055, phi3=0.01, phi4=0.01,
            mu1t=0.017, mu2t=0.017, mu3t=0.017, mu4t=0.017,
            muBt=0.017, muCt=0.017, muDt=0.017, muEt=0.017,
            k1t=0.55, k2t=0.55, k3t=0.1, k4t=0.1,
            p1t=0.15, p2t=0.15, p3t=0.1, p4t=0.1,
            gamma1t=0.01, gamma2t=0.01, gamma3t=0.01, gamma4t=0.01,
            phi1t=0.050, phi2t=0.055, phi3t=0.015, phi4t=0.01,
            )

for i in range(10):
    solver = RungeKutta4(model)
    solver.set_initial_condition([V10, B0, C0, V20, D0, E0])
    dt = 0.2  # Time step
    T = 168  # Total time
    n = int(T / dt)  # Number of time steps
    time_points = np.linspace(0, T, n)  # Vector times
    sqrtdt = np.sqrt(dt)
    u, t = solver.solve(time_points)
    V1 = u[:, 0];
    B = u[:, 1];
    C = u[:, 2];
    V2 = u[:, 3];
    D = u[:, 4];
    E = u[:, 5]

    axs[1, 1].title.set_text('S0101')
    axs[1, 1].plot(t, V1, label="Sim.Abs", color="yellow", alpha=0.1, zorder=1, linewidth=0.5)
    axs[1, 1].plot(t, B, label="Sim.Dem", color="blue", alpha=0.1, zorder=1, linewidth=0.5)
    axs[1, 1].plot(t, C, label="Sim.Rep", color="red", alpha=0.1, zorder=1, linewidth=0.5)
    axs[1, 1].scatter(t_us, np.array(Total_Abstention) * 1000000, label="Abs", color="yellow", s=4, zorder=2, edgecolors= "black", linewidth=0.1)
    axs[1, 1].scatter(t_us, np.array(Total_US_dem) * 1000000, label="Dem", color="blue", s=4, zorder=2, edgecolors= "black", linewidth=0.1)
    axs[1, 1].scatter(t_us, np.array(Total_US_rep) * 1000000, label="Rep", color="red", s=4, zorder=2, edgecolors= "black", linewidth=0.1)
    #axs[1, 1].set_xlabel('Time in years')
    #axs[1, 1].set_ylabel('Number of agents')
    axs[1,1].set_yticklabels([])
    axs[1, 1].set_ylim(0, 200000000)
    x = np.arange(0, 169, 42)
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels([i + 1932 for i in x])
    axs[1, 1].patch.set_facecolor('blue')
    axs[1, 1].patch.set_alpha(0.05)
    axs[1, 1].set_xticklabels([])

# Sixth simulation
#Initial conditions country 1 (US)
V10 = 34650000
B0 = 19291265
C0 = 19291265
#C0 = 19000000
#Initial conditions country 2 # At t0 = 1932, for agents outside the U.S., it is assumed that
#political tendencies are evenly split at the start of the simulation, with one-third non-partisan, one-third
# pro-Democrats, and one-third pro-Republicans.
V20 = 50000000
D0 = 50000000
E0 = 50000000
model = VBC(r1=0.02,r2=0.02,
            mu1=0.017, mu2=0.017, mu3=0.017, mu4=0.017,
            muB= 0.017, muC= 0.017, muD=0.017, muE=0.017,
            k1= 0.55, k2= 0.55, k3=0.1, k4=0.1,
            p1= 0.15, p2= 0.15, p3=0.1, p4=0.1,
            gamma1= 0.01, gamma2= 0.01, gamma3=0.01, gamma4=0.01,
            phi1= 0.050, phi2= 0.050, phi3=0.01, phi4=0.01,
            mu1t=0.017, mu2t=0.017, mu3t=0.017, mu4t=0.017,
            muBt=0.017, muCt=0.017, muDt=0.017, muEt=0.017,
            k1t=0.55, k2t=0.55, k3t=0.1, k4t=0.1,
            p1t=0.15, p2t=0.15, p3t=0.1, p4t=0.1,
            gamma1t=0.01, gamma2t=0.01, gamma3t=0.01, gamma4t=0.01,
            phi1t=0.050, phi2t=0.050, phi3t=0.015, phi4t=0.01,
            )

for i in range(10):
    solver = RungeKutta4(model)
    solver.set_initial_condition([V10, B0, C0, V20, D0, E0])
    dt = 0.2  # Time step
    T = 168  # Total time
    n = int(T / dt)  # Number of time steps
    time_points = np.linspace(0, T, n)  # Vector times
    sqrtdt = np.sqrt(dt)
    u, t = solver.solve(time_points)
    V1 = u[:, 0];
    B = u[:, 1];
    C = u[:, 2];
    V2 = u[:, 3];
    D = u[:, 4];
    E = u[:, 5]

    axs[1, 2].title.set_text('S1001')
    axs[1, 2].plot(t, V1, label="Sim.Abs", color="yellow", alpha=0.1, zorder=1, linewidth=0.5)
    axs[1, 2].plot(t, B, label="Sim.Dem", color="blue", alpha=0.1, zorder=1, linewidth=0.5)
    axs[1, 2].plot(t, C, label="Sim.Rep", color="red", alpha=0.1, zorder=1, linewidth=0.5)
    axs[1, 2].scatter(t_us, np.array(Total_Abstention) * 1000000, label="Abs", color="yellow", s=4, zorder=2, edgecolors= "black", linewidth=0.1)
    axs[1, 2].scatter(t_us, np.array(Total_US_dem) * 1000000, label="Dem", color="blue", s=4, zorder=2, edgecolors= "black", linewidth=0.1)
    axs[1, 2].scatter(t_us, np.array(Total_US_rep) * 1000000, label="Rep", color="red", s=4, zorder=2, edgecolors= "black", linewidth=0.1)
    # axs[1, 2].set_xlabel('Time in years')
    # axs[1, 2].set_ylabel('Number of agents')
    axs[1, 2].set_yticklabels([])
    axs[1, 2].set_ylim(0, 200000000)
    x = np.arange(0, 169, 42)
    axs[1, 2].set_xticks(x)
    axs[1, 2].set_xticklabels([i + 1932 for i in x])
    axs[1, 2].patch.set_facecolor('blue')
    axs[1, 2].patch.set_alpha(0.05)
    axs[1, 2].set_xticklabels([])


# Seventh simulation
#Initial conditions country 1 (US)
V10 = 34650000
# B0 = 22821277
# C0 = 15761254
B0 = 22000000
C0 = 16000000
#C0 = 19000000
#Initial conditions country 2 # At t0 = 1932, for agents outside the U.S., it is assumed that
#political tendencies are evenly split at the start of the simulation, with one-third non-partisan, one-third
# pro-Democrats, and one-third pro-Republicans.
V20 = 50000000
D0 = 50000000
E0 = 50000000
model = VBC(r1=0.02,r2=0.02,
            mu1=0.017, mu2=0.017, mu3=0.017, mu4=0.017,
            muB= 0.017, muC= 0.017, muD=0.017, muE=0.017,
            k1= 0.55, k2= 0.55, k3=0.1, k4=0.1,
            p1= 0.15, p2= 0.15, p3=0.1, p4=0.1,
            gamma1= 0.01, gamma2= 0.01, gamma3=0.01, gamma4=0.01,
            phi1= 0.050, phi2= 0.050, phi3=0.01, phi4=0.01,
            mu1t=0.017, mu2t=0.017, mu3t=0.017, mu4t=0.017,
            muBt=0.017, muCt=0.017, muDt=0.017, muEt=0.017,
            k1t=0.55, k2t=0.55, k3t=0.1, k4t=0.1,
            p1t=0.15, p2t=0.15, p3t=0.1, p4t=0.1,
            gamma1t=0.01, gamma2t=0.01, gamma3t=0.01, gamma4t=0.01,
            phi1t=0.050, phi2t=0.050, phi3t=0.01, phi4t=0.015,
            )


#number of simulations
for i in range(10):
    solver = RungeKutta4(model)
    solver.set_initial_condition([V10, B0, C0, V20, D0, E0])
    dt = 0.2  # Time step
    T = 168  # Total time
    n = int(T / dt)  # Number of time steps
    time_points = np.linspace(0, T, n)  # Vector times
    sqrtdt = np.sqrt(dt)
    u, t = solver.solve(time_points)
    V1 = u[:, 0];
    B = u[:, 1];
    C = u[:, 2];
    V2 = u[:, 3];
    D = u[:, 4];
    E = u[:, 5]

    axs[2, 0].title.set_text('S0010')
    axs[2, 0].plot(t, V1, label="Sim.Abs", color="yellow", alpha=0.1, zorder=1, linewidth=0.5)
    axs[2, 0].plot(t, B, label="Sim.Dem", color="blue", alpha=0.1, zorder=1, linewidth=0.5)
    axs[2, 0].plot(t, C, label="Sim.Rep", color="red", alpha=0.1, zorder=1, linewidth=0.5)
    axs[2, 0].scatter(t_us, np.array(Total_Abstention) * 1000000, label="Abs", color="yellow", s=4, zorder=2,
                      edgecolors= "black", linewidth=0.1)
    axs[2, 0].scatter(t_us, np.array(Total_US_dem) * 1000000, label="Dem", color="blue", s=4, zorder=2, edgecolors= "black", linewidth=0.1)
    axs[2, 0].scatter(t_us, np.array(Total_US_rep) * 1000000, label="Rep", color="red", s=4, zorder=2, edgecolors= "black", linewidth=0.1)
    #axs[2, 0].set_xlabel('Time in years')
    #axs[1, 0].set_ylabel('Number of agents')
    #axs[1, 0].set_xticklabels([])
    axs[2, 0].set_ylim(0, 200000000)
    x = np.arange(0, 169, 42)
    axs[2, 0].set_xticks(x)
    axs[2, 0].set_xticklabels([i + 1932 for i in x])
    axs[2, 0].patch.set_facecolor('red')
    axs[2, 0].patch.set_alpha(0.05)
    axs[2, 0].set_xticklabels([])



# Eighth simulation
#Initial conditions country 1 (US)
V10 = 34650000
# B0 = 22821277
# C0 = 15761254
B0 = 22000000
C0 = 16000000
#C0 = 19000000
#Initial conditions country 2 # At t0 = 1932, for agents outside the U.S., it is assumed that
#political tendencies are evenly split at the start of the simulation, with one-third non-partisan, one-third
# pro-Democrats, and one-third pro-Republicans.
V20 = 50000000
D0 = 50000000
E0 = 50000000
model = VBC(r1=0.02,r2=0.02,
            mu1=0.017, mu2=0.017, mu3=0.017, mu4=0.017,
            muB= 0.017, muC= 0.017, muD=0.017, muE=0.017,
            k1= 0.55, k2= 0.55, k3=0.1, k4=0.1,
            p1= 0.15, p2= 0.15, p3=0.1, p4=0.1,
            gamma1= 0.01, gamma2= 0.01, gamma3=0.01, gamma4=0.01,
            phi1= 0.050, phi2= 0.055, phi3=0.01, phi4=0.01,
            mu1t=0.017, mu2t=0.017, mu3t=0.017, mu4t=0.017,
            muBt=0.017, muCt=0.017, muDt=0.017, muEt=0.017,
            k1t=0.55, k2t=0.55, k3t=0.1, k4t=0.1,
            p1t=0.15, p2t=0.15, p3t=0.1, p4t=0.1,
            gamma1t=0.01, gamma2t=0.01, gamma3t=0.01, gamma4t=0.01,
            phi1t=0.050, phi2t=0.055, phi3t=0.01, phi4t=0.015,
            )


for i in range(10):
    solver = RungeKutta4(model)
    solver.set_initial_condition([V10, B0, C0, V20, D0, E0])
    dt = 0.2  # Time step
    T = 168  # Total time
    n = int(T / dt)  # Number of time steps
    time_points = np.linspace(0, T, n)  # Vector times
    sqrtdt = np.sqrt(dt)
    u, t = solver.solve(time_points)
    V1 = u[:, 0];
    B = u[:, 1];
    C = u[:, 2];
    V2 = u[:, 3];
    D = u[:, 4];
    E = u[:, 5]

    axs[2, 1].title.set_text('S0110')
    axs[2, 1].plot(t, V1, label="Sim.Abs", color="yellow", alpha=0.1, zorder=1, linewidth=0.5)
    axs[2, 1].plot(t, B, label="Sim.Dem", color="blue", alpha=0.1, zorder=1, linewidth=0.5)
    axs[2, 1].plot(t, C, label="Sim.Rep", color="red", alpha=0.1, zorder=1, linewidth=0.5)
    axs[2, 1].scatter(t_us, np.array(Total_Abstention) * 1000000, label="Abs", color="yellow", s=4, zorder=2, edgecolors= "black", linewidth=0.1)
    axs[2, 1].scatter(t_us, np.array(Total_US_dem) * 1000000, label="Dem", color="blue", s=4, zorder=2, edgecolors= "black", linewidth=0.1)
    axs[2, 1].scatter(t_us, np.array(Total_US_rep) * 1000000, label="Rep", color="red", s=4, zorder=2, edgecolors= "black", linewidth=0.1)
    #axs[2, 1].set_xlabel('Time in years')
    #axs[2, 1].set_ylabel('Number of agents')
    axs[2, 1].set_yticklabels([])
    x = np.arange(0, 169, 42)
    axs[2, 1].set_xticks(x)
    axs[2, 1].set_xticklabels([i + 1932 for i in x])
    axs[2, 1].patch.set_facecolor('red')
    axs[2, 1].patch.set_alpha(0.05)
    axs[2, 1].set_xticklabels([])

# Nineth simulation
#Initial conditions country 1 (US)
V10 = 34650000
B0 = 19291265
C0 = 19291265
#C0 = 19000000
#Initial conditions country 2 # At t0 = 1932, for agents outside the U.S., it is assumed that
#political tendencies are evenly split at the start of the simulation, with one-third non-partisan, one-third
# pro-Democrats, and one-third pro-Republicans.
V20 = 50000000
D0 = 50000000
E0 = 50000000
model = VBC(r1=0.02,r2=0.02,
            mu1=0.017, mu2=0.017, mu3=0.017, mu4=0.017,
            muB= 0.017, muC= 0.017, muD=0.017, muE=0.017,
            k1= 0.55, k2= 0.55, k3=0.1, k4=0.1,
            p1= 0.15, p2= 0.15, p3=0.1, p4=0.1,
            gamma1= 0.01, gamma2= 0.01, gamma3=0.01, gamma4=0.01,
            phi1= 0.050, phi2= 0.050, phi3=0.01, phi4=0.01,
            mu1t=0.017, mu2t=0.017, mu3t=0.017, mu4t=0.017,
            muBt=0.017, muCt=0.017, muDt=0.017, muEt=0.017,
            k1t=0.55, k2t=0.55, k3t=0.1, k4t=0.1,
            p1t=0.15, p2t=0.15, p3t=0.1, p4t=0.1,
            gamma1t=0.01, gamma2t=0.01, gamma3t=0.01, gamma4t=0.01,
            phi1t=0.050, phi2t=0.050, phi3t=0.01, phi4t=0.015,
            )

for i in range(10):
    solver = RungeKutta4(model)
    solver.set_initial_condition([V10, B0, C0, V20, D0, E0])
    dt = 0.2  # Time step
    T = 168  # Total time
    n = int(T / dt)  # Number of time steps
    time_points = np.linspace(0, T, n)  # Vector times
    sqrtdt = np.sqrt(dt)
    u, t = solver.solve(time_points)
    V1 = u[:, 0];
    B = u[:, 1];
    C = u[:, 2];
    V2 = u[:, 3];
    D = u[:, 4];
    E = u[:, 5]

    axs[2, 2].title.set_text('S1010')
    axs[2, 2].plot(t, V1, label="Sim.Abs", color="yellow", alpha=0.1, zorder=1, linewidth=0.5)
    axs[2, 2].plot(t, B, label="Sim.Dem", color="blue", alpha=0.1, zorder=1, linewidth=0.5)
    axs[2, 2].plot(t, C, label="Sim.Rep", color="red", alpha=0.1, zorder=1, linewidth=0.5)
    axs[2, 2].scatter(t_us, np.array(Total_Abstention) * 1000000, label="Abs", color="yellow", s=4, zorder=2, edgecolors= "black", linewidth=0.1)
    axs[2, 2].scatter(t_us, np.array(Total_US_dem) * 1000000, label="Dem", color="blue", s=4, zorder=2, edgecolors= "black", linewidth=0.1)
    axs[2, 2].scatter(t_us, np.array(Total_US_rep) * 1000000, label="Rep", color="red", s=4, zorder=2, edgecolors= "black", linewidth=0.1)
    # axs[2, 2].set_xlabel('Time in years')
    # axs[2, 2].set_ylabel('Number of agents')
    axs[2, 2].set_yticklabels([])
    axs[2, 2].set_ylim(0, 200000000)
    x = np.arange(0, 169, 42)
    axs[2, 2].set_xticks(x)
    axs[2, 2].set_xticklabels([i + 1932 for i in x])
    axs[2, 2].patch.set_facecolor('red')
    axs[2, 2].patch.set_alpha(0.05)
    axs[2, 2].set_xticklabels([])


# Tenth simulation
#Initial conditions country 1 (US)
V10 = 34650000
# B0 = 22821277
# C0 = 15761254
B0 = 22000000
C0 = 16000000
#C0 = 19000000
#Initial conditions country 2 # At t0 = 1932, for agents outside the U.S., it is assumed that
#political tendencies are evenly split at the start of the simulation, with one-third non-partisan, one-third
# pro-Democrats, and one-third pro-Republicans.
V20 = 50000000
D0 = 50000000
E0 = 50000000
model = VBC(r1=0.02,r2=0.02,
            mu1=0.017, mu2=0.017, mu3=0.017, mu4=0.017,
            muB= 0.017, muC= 0.017, muD=0.017, muE=0.017,
            k1= 0.55, k2= 0.55, k3=0.1, k4=0.1,
            p1= 0.15, p2= 0.15, p3=0.1, p4=0.1,
            gamma1= 0.01, gamma2= 0.01, gamma3=0.01, gamma4=0.01,
            phi1= 0.050, phi2= 0.050, phi3=0.01, phi4=0.01,
            mu1t=0.017, mu2t=0.017, mu3t=0.017, mu4t=0.017,
            muBt=0.017, muCt=0.017, muDt=0.017, muEt=0.017,
            k1t=0.55, k2t=0.55, k3t=0.1, k4t=0.1,
            p1t=0.15, p2t=0.15, p3t=0.1, p4t=0.1,
            gamma1t=0.015, gamma2t=0.015, gamma3t=0.015, gamma4t=0.015,
            phi1t=0.050, phi2t=0.050, phi3t=0.01, phi4t=0.01,
            )


#number of simulations
for i in range(10):
    solver = RungeKutta4(model)
    solver.set_initial_condition([V10, B0, C0, V20, D0, E0])
    dt = 0.2  # Time step
    T = 168  # Total time
    n = int(T / dt)  # Number of time steps
    time_points = np.linspace(0, T, n)  # Vector times
    sqrtdt = np.sqrt(dt)
    u, t = solver.solve(time_points)
    V1 = u[:, 0];
    B = u[:, 1];
    C = u[:, 2];
    V2 = u[:, 3];
    D = u[:, 4];
    E = u[:, 5]

    axs[3, 0].title.set_text('S0011')
    axs[3, 0].plot(t, V1, label="Sim.Abs", color="yellow", alpha=0.1, zorder=1, linewidth=0.5)
    axs[3, 0].plot(t, B, label="Sim.Dem", color="blue", alpha=0.1, zorder=1, linewidth=0.5)
    axs[3, 0].plot(t, C, label="Sim.Rep", color="red", alpha=0.1, zorder=1, linewidth=0.5)
    axs[3, 0].scatter(t_us, np.array(Total_Abstention) * 1000000, label="Abs", color="yellow", s=4, zorder=2,
                      edgecolors= "black", linewidth=0.1)
    axs[3, 0].scatter(t_us, np.array(Total_US_dem) * 1000000, label="Dem", color="blue", s=4, zorder=2, edgecolors=
    "black", linewidth=0.1)
    axs[3, 0].scatter(t_us, np.array(Total_US_rep) * 1000000, label="Rep", color="red", s=4, zorder=2, edgecolors=
    "black", linewidth=0.1)
    #axs[2, 0].set_xlabel('Time in years')
    #axs[1, 0].set_ylabel('Number of agents')
    #axs[1, 0].set_xticklabels([])
    axs[3, 0].set_ylim(0, 200000000)
    x = np.arange(0, 169, 42)
    axs[3,0].set_xticks(x)
    axs[3,0].set_xticklabels([i + 1932 for i in x])
    axs[3, 0].patch.set_facecolor('yellow')
    axs[3, 0].patch.set_alpha(0.05)



# Eleventh simulation
#Initial conditions country 1 (US)
V10 = 34650000
# B0 = 22821277
# C0 = 15761254
B0 = 22000000
C0 = 16000000
#C0 = 19000000
#Initial conditions country 2 # At t0 = 1932, for agents outside the U.S., it is assumed that
#political tendencies are evenly split at the start of the simulation, with one-third non-partisan, one-third
# pro-Democrats, and one-third pro-Republicans.
V20 = 50000000
D0 = 50000000
E0 = 50000000
model = VBC(r1=0.02,r2=0.02,
            mu1=0.017, mu2=0.017, mu3=0.017, mu4=0.017,
            muB= 0.017, muC= 0.017, muD=0.017, muE=0.017,
            k1= 0.55, k2= 0.55, k3=0.1, k4=0.1,
            p1= 0.15, p2= 0.15, p3=0.1, p4=0.1,
            gamma1= 0.01, gamma2= 0.01, gamma3=0.01, gamma4=0.01,
            phi1= 0.050, phi2= 0.055, phi3=0.01, phi4=0.01,
            mu1t=0.017, mu2t=0.017, mu3t=0.017, mu4t=0.017,
            muBt=0.017, muCt=0.017, muDt=0.017, muEt=0.017,
            k1t=0.55, k2t=0.55, k3t=0.1, k4t=0.1,
            p1t=0.15, p2t=0.15, p3t=0.1, p4t=0.1,
            gamma1t=0.015, gamma2t=0.015, gamma3t=0.015, gamma4t=0.015,
            phi1t=0.050, phi2t=0.055, phi3t=0.01, phi4t=0.01,
            )

for i in range(10):
    solver = RungeKutta4(model)
    solver.set_initial_condition([V10, B0, C0, V20, D0, E0])
    dt = 0.2  # Time step
    T = 168  # Total time
    n = int(T / dt)  # Number of time steps
    time_points = np.linspace(0, T, n)  # Vector times
    sqrtdt = np.sqrt(dt)
    u, t = solver.solve(time_points)
    V1 = u[:, 0];
    B = u[:, 1];
    C = u[:, 2];
    V2 = u[:, 3];
    D = u[:, 4];
    E = u[:, 5]

    axs[3, 1].title.set_text('S0111')
    axs[3, 1].plot(t, V1, label="Sim.Abs", color="yellow", alpha=0.1, zorder=1, linewidth=0.5)
    axs[3, 1].plot(t, B, label="Sim.Dem", color="blue", alpha=0.1, zorder=1, linewidth=0.5)
    axs[3, 1].plot(t, C, label="Sim.Rep", color="red", alpha=0.1, zorder=1, linewidth=0.5)
    axs[3, 1].scatter(t_us, np.array(Total_Abstention) * 1000000, label="Abs", color="yellow", s=4, zorder=2,
                      edgecolors= "black", linewidth=0.1)
    axs[3, 1].scatter(t_us, np.array(Total_US_dem) * 1000000, label="Dem", color="blue", s=4, zorder=2, edgecolors=
    "black", linewidth=0.1)
    axs[3, 1].scatter(t_us, np.array(Total_US_rep) * 1000000, label="Rep", color="red", s=4, zorder=2, edgecolors=
    "black", linewidth=0.1)
    axs[3, 1].set_xlabel('Time in years', fontsize=12)
    #axs[2, 1].set_ylabel('Number of agents')
    axs[3, 1].set_yticklabels([])
    axs[3, 1].set_ylim(0, 200000000)
    x = np.arange(0, 169, 42)
    axs[3, 1].set_xticks(x)
    axs[3, 1].set_xticklabels([i + 1932 for i in x])
    axs[3, 1].patch.set_facecolor('yellow')
    axs[3, 1].patch.set_alpha(0.05)

# Twelveth simulation
#Initial conditions country 1 (US)
V10 = 34650000
B0 = 19291265
C0 = 19291265
#C0 = 19000000
#Initial conditions country 2 # At t0 = 1932, for agents outside the U.S., it is assumed that
#political tendencies are evenly split at the start of the simulation, with one-third non-partisan, one-third
# pro-Democrats, and one-third pro-Republicans.
V20 = 50000000
D0 = 50000000
E0 = 50000000
model = VBC(r1=0.02,r2=0.02,
            mu1=0.017, mu2=0.017, mu3=0.017, mu4=0.017,
            muB= 0.017, muC= 0.017, muD=0.017, muE=0.017,
            k1= 0.55, k2= 0.55, k3=0.1, k4=0.1,
            p1= 0.15, p2= 0.15, p3=0.1, p4=0.1,
            gamma1= 0.01, gamma2= 0.01, gamma3=0.01, gamma4=0.01,
            phi1= 0.050, phi2= 0.050, phi3=0.01, phi4=0.01,
            mu1t=0.017, mu2t=0.017, mu3t=0.017, mu4t=0.017,
            muBt=0.017, muCt=0.017, muDt=0.017, muEt=0.017,
            k1t=0.55, k2t=0.55, k3t=0.1, k4t=0.1,
            p1t=0.15, p2t=0.15, p3t=0.1, p4t=0.1,
            gamma1t=0.015, gamma2t=0.015, gamma3t=0.015, gamma4t=0.015,
            phi1t=0.050, phi2t=0.050, phi3t=0.01, phi4t=0.01,
            )


for i in range(10):
    solver = RungeKutta4(model)
    solver.set_initial_condition([V10, B0, C0, V20, D0, E0])
    dt = 0.2  # Time step
    T = 168  # Total time
    n = int(T / dt)  # Number of time steps
    time_points = np.linspace(0, T, n)  # Vector times
    sqrtdt = np.sqrt(dt)
    u, t = solver.solve(time_points)
    V1 = u[:, 0];
    B = u[:, 1];
    C = u[:, 2];
    V2 = u[:, 3];
    D = u[:, 4];
    E = u[:, 5]

    axs[3, 2].title.set_text('S1011')
    axs[3, 2].plot(t, V1, label="Sim.Abs", color="yellow", alpha=0.1, zorder=1, linewidth=0.5)
    axs[3, 2].plot(t, B, label="Sim.Dem", color="blue", alpha=0.1, zorder=1, linewidth=0.5)
    axs[3, 2].plot(t, C, label="Sim.Rep", color="red", alpha=0.1, zorder=1, linewidth=0.5)
    axs[3, 2].scatter(t_us, np.array(Total_Abstention) * 1000000, label="Abs", color="yellow", s=4, zorder=2,
                      edgecolors= "black", linewidth=0.1)
    axs[3, 2].scatter(t_us, np.array(Total_US_dem) * 1000000, label="Dem", color="blue", s=4, zorder=2, edgecolors=
    "black", linewidth=0.1)
    axs[3, 2].scatter(t_us, np.array(Total_US_rep) * 1000000, label="Rep", color="red", s=4, zorder=2, edgecolors=
    "black", linewidth=0.1)
    # axs[2, 2].set_xlabel('Time in years')
    # axs[2, 2].set_ylabel('Number of agents')
    axs[3, 2].set_yticklabels([])
    axs[3, 2].set_ylim(0, 200000000)
    x = np.arange(0, 169, 42)
    axs[3, 2].set_xticks(x)
    axs[3, 2].set_xticklabels([i + 1932 for i in x])
    axs[3, 2].patch.set_facecolor('yellow')
    axs[3, 2].patch.set_alpha(0.05)


# handles, labels = axs[1, 1].get_legend_handles_labels()
# handle_list, label_list = [], []
# for handle, label in zip(handles, labels):
#         if label not in label_list:
#             handle_list.append(handle)
#             label_list.append(label)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
leg = axs[3,1].legend(by_label.values(), by_label.keys(), ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.3))

#leg = axs[3,1].legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.3))
for lh in leg.legendHandles:
    lh.set_alpha(1)

plt.tight_layout()
plt.show()
#fig.set_size_inches(10, 12)
#fig.set_size_inches(10,8)
fig.set_size_inches(8,8)
fig.subplots_adjust(bottom=0.15)
#fig.text(0.6, 0.1, 'Time in years', ha='center', size= 1.5)
fig.text(0.004, 0.55, 'Supporters', va='center', rotation='vertical', fontsize=12)
fig.savefig('US_results_2100_3.pdf')
