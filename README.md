# Modelling the dynamics of cross-border ideological competition
Mathematical model for the cross-border spread of two ideologies by using an epidemiological approach. 

The ODEsolver.py module contains a number of classes that implement numerical methods for solving ordinary differential equations. This module borrows heavily from the work of Joakim Sundnes, 
https://github.com/sundnes/solving_odes_in_python

The Model.py file contains the epidemiological implementation of cross-border ideological competition. The model is implemented as a class and solved numerically by the Runge-Kutta method. Running the script yields results from four simulations.

Model parameters can be manipulated for each simulation in the corresponding section. For example, for the first simulation we have:
```
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
            gamma1= 0.01, gamma2= 0.01, gamma3=0.01, gamma4=0.01,
            phi1= 0.02, phi2= 0.02, phi3=0.02, phi4=0.02)
```
