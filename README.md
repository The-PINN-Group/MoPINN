# MoPINN
Common Pitfalls to avoid while using multi-objective optimization in machine learning

# Install
1. create conda environment
2. install pip via
'''
conda install pip
'''
3. install packages (from within the folder) via
'''
pip install -r requirements.txt
'''

# 4 Experiments
All experiments create Pareto fronts using noisy data and the PDE loss (logistic equation or heat equation)

## EA_NSGA_logistic.py
Evolutionary Algorithm using NSGA-II to create a Pareto front.

## MGDA_logistic.ipynb
MGDA training a bunch of models, selecting the non-dominated points at the end of training for the Pareto front.

## WS_heat.ipynb
Weighted Sum training (heat equation) of 20 models with different alpha-values (weight given to physics and data loss).

## WS_logistic.ipynb
Weighted Sum training (logistic equation) of 20 models with different alpha-values (weight given to physics and data loss).

