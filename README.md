# SDP-MILP4OPF
A global optimization algorithm for the ACOPF problem

# Test instances

The ACOPF instances are taken from the library PGLib (https://github.com/power-grid-lib/pglib-opf), which is maintained by the IEEE PES Task Force on Benchmarks for Validation of Emerging Power System Algorithms.


# Programming language, installation and dependencies

dualACOPFsolver is implemented in Python3. To run this code, you have to clone this repository. The required packages are:
- numpy
- scipy
- pandas
- docplex (with CPLEX license)
- mosek (with MOSEK license)
- chompack
- cvxopt 
- progress


You also have to replace the empty pglib-opf folder by a clone of https://github.com/power-grid-lib/pglib-opf, so as to have the test instances.


# Running our numerical experiments

Executing 
```
python3 main_typ.py
```

and

```
python3 main_api.py
```

will run the numerical experiments presented in the paper "A. Oustry, AC Optimal Power Flow: a Conic Programming relaxation and an iterative MILP scheme for Global Optimization, Open Journal of Mathematical Optimization, 2022".  

---------------------------------------------------------------------------------------
# Affiliations and sponsor

Researchers affiliated with

(o) LIX CNRS, École polytechnique, Institut Polytechnique de Paris, 91128, Palaiseau, France 

(o) École des Ponts, 77455 Marne-La-Vallée

---------------------------------------------------------------------------------------

Sponsored by Réseau de transport d’électricité, 92073 La Défense, France




