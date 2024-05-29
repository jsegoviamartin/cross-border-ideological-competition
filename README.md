# Modelling the dynamics of cross-border ideological competition
Mathematical model for the cross-border spread of two ideologies, as described in:

* Research article (open access):
  
Segovia-Martin J, Rivero Ó (2024) Cross-border political competition. PLoS ONE 19(5): e0297731. https://doi.org/10.1371/journal.pone.0297731

* Preprint:
  
Segovia-Martin, J. (2022). Modelling the dynamics of cross-border ideological competition. arXiv preprint arXiv:2205.06010.
arXiv preprint: https://arxiv.org/abs/2205.06010

## Citation
If you use this work, please cite the following:

BibTeX (compatible with BibDesk, LaTeX)
```
@article{10.1371/journal.pone.0297731,
    doi = {10.1371/journal.pone.0297731},
    author = {Segovia-Martin, Jose AND Rivero, Óscar},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {Cross-border political competition},
    year = {2024},
    month = {05},
    volume = {19},
    url = {https://doi.org/10.1371/journal.pone.0297731},
    pages = {1-26},
    abstract = {Individuals are increasingly exposed to news and opinion from beyond national borders. This news and opinion are often concentrated in clusters of ideological homophily, such as political parties, factions, or interest groups. But how does exposure to cross-border information affect the diffusion of ideas across national and ideological borders? Here, we develop a non-linear mathematical model for the cross-border spread of two ideologies. First, we describe the standard deterministic model where the populations of each country are assumed to be constant and homogeneously mixed. We solve the system of differential equations numerically by the Runge-Kutta method and show how small changes in the influence of a minority ideology can trigger shifts in the global political equilibrium. Second, we simulate recruitment as a stochastic differential process for each political affiliation and fit model solutions to population growth rates and voting populations in US presidential elections from 1932 to 2020. We also project the dynamics of several possible scenarios from 2020 to the end of the century. We show that cross-border influence plays a fundamental role in determining election outcomes. An increase in foreign support for a national party’s ideas could change the election outcome, independent of domestic recruitment capacity. One key finding of our study suggests that voter turnout in the US will grow at a faster rate than non-voters in the coming decades. This trend is attributed to the enhanced recruitment capabilities of both major parties among non-partisans over time, making political disaffection less prominent. This phenomenon holds true across all simulated scenarios.},
    number = {5},

}
```
## Scripts

The ODEsolver.py module contains a number of classes that implement numerical methods for solving ordinary differential equations. This module borrows heavily from the work of Joakim Sundnes, 
https://github.com/sundnes/solving_odes_in_python

Three alternative implementations for the RungeKutta4 class in RungeKutta4_List_Comprehensions.py, RungeKutta4_Vectorized_Approach.py and RungeKutta4_explicit_handling.py.

The Model.py file contains the non-lineal implementation of cross-border ideological competition. The model is implemented as a class and solved numerically by the Runge-Kutta method. Running the script yields results from four simulations.

## Requirements

Make sure you have the following Python packages installed:
- numpy
- matplotlib

You can install them using pip:
```bash
pip install numpy matplotlib
```
## Running Simulations from an IDE using Model.py

1. Open the Model.py script in your preferred IDE (e.g., PyCharm, VSCode).

2. Modify the initial conditions and parameters directly in the script as needed. Model parameters can be manipulated for each simulation in the corresponding section. Here is a sample configuration:
   
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

3. Run the the script from your IDE. The results will be printed to the console, and plots will be generated.

## Running Simulations from the Command Line using run_model.py

1. Open a terminal or command prompt.

2. Navigate to the script directory to where the run_model.py script is located.
```
cd path/to/your/script
```

3. Run the script with the desired initial conditions and parameters (unspecified parameters will use default values provided in the script). Here is an example command:
```
python run_model.py --V10 1500 --B0 1500 --C0 1500 --V20 2000 --D0 2000 --E0 2000 --k1 0.6 --k2 0.7
```
4. The results will be printed to the console.


