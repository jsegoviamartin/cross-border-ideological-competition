# Modelling the dynamics of cross-border ideological competition
Mathematical model for the cross-border spread of two ideologies by using an epidemiological approach, as described in:

Segovia-Martin, J. (2022). Modelling the dynamics of cross-border ideological competition. arXiv preprint arXiv:2205.06010.
arXiv preprint: https://arxiv.org/abs/2205.06010

## Citation
If you use this model in your work, please cite the following:

Bib TeX of the preprint
```
@article{segovia2022modelling,
  title={Modelling the dynamics of cross-border ideological competition},
  author={Segovia-Martin, Jose},
  journal={arXiv preprint arXiv:2205.06010},
  year={2022}
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


