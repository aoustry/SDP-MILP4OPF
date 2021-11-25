# SDP-MILP4OPF
A global optimization algorithm for the ACOPF problem

# Programming language and dependencies

dualACOPFsolver is implemented in Python3. The required packages are:
- numpy
- scipy
- pandas
- docplex
- mosek
- chompack
- cvxopt 
- progressbar

# Test instances

The ACOPF instances are taken from the library PGLib (https://github.com/power-grid-lib/pglib-opf), which is maintained by the IEEE PES Task Force on Benchmarks for Validation of Emerging Power System Algorithms.

# Numerical experiments

Executing 
```
python3 main_typ.py
```

and

```
python3 main_api.py
```

will run the numerical experiments presented in the paper "A. Oustry, AC Optimal Power Flow: a strengthened SDP relaxation and an iterative MILP scheme for global optimization" (submitted).  

---------------------------------------------------------------------------------------
# Affiliations and sponsor

Researchers affiliated with

(o) LIX CNRS, École polytechnique, Institut Polytechnique de Paris, 91128, Palaiseau, France 

(o) École des Ponts, 77455 Marne-La-Vallée

---------------------------------------------------------------------------------------

Sponsored by Réseau de transport d’électricité, 92073 La Défense, France




