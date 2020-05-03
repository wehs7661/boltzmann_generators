Boltzmann Generators
==============================

## Description
This is a repository for the final project of the course Computational Statistical Physics (PHYS 7810) at CU Boulder, of which the goal is to develop and apply [Boltzmann generators](https://science.sciencemag.org/content/365/6457/eaaw1147) to different molecular systems. The project paper can be found in the folder `Project`. Specifically, this repository includes the following directories:
- `Project`: A folder containing the project paper, presentation slides, and pictures for the final proejct.
- `References`: A folder containing the most important journal papers relevant to our project.
- `Notebooks`: A folder containing severl jupyter notebooks in a tutorial style which implement Boltzmann generators and applies them to different systems of interest. From the introductory tutorial of PyTorch and the simplest toy model, to a more advanced molecular system, these notebooks include:
  - `PyTorch Introduction.ipynb`: A jupyter notebook adapted from a tutorial generously provided by the course [CSE446: Machine Learning at Univeristy of Washington](https://courses.cs.washington.edu/courses/cse446/19au/section9.html).
  - `Double-well Potential.ipynb`: A notebook which implements the architecture of Boltzmann generators and applies them to the simplest toy model: double-well potential.
  - `Mueller Brown Potential.ipynb`: A notebook which applies Boltzmann generators to the slightly more complex Mueller potential, which is characteristic of three energy minima and a more complicated reaction coordinate.
  - `Dimer-Simulation.ipynb`: A notebook which applies Boltzmann generators to a dimer in Lennard-Jones bath.
- `Library`: A folder of software implementing Boltzmann generators that can be imported by the jupyter notebooks in the folder `Notebooks`.

## Contributions
- Both authors contributed equally to the project.
- Wei-Tse Hsu
  - Developed codes for performing Monte Carlo simulations.
  - Developed codes for building and training a Boltzmann generators and corresponding data analysis. (`generator.py`, `training.py`, `density estimator.py`, `visuals.py`.)
  - Applied Boltzmann generators to the double-well potential (`Double-well Potential.ipynb`).
  - Presentation slides (page 1 to page 20).
  - Project paper: sections relevant to the work above.

- Lenny Fobe
  - Developed codes for performing molecular dynamics simulation.
  - Applied Boltzmann generators to the Muller Brown potential and the dimer in the Lennard-Jones bath. (`Mueller Brown Potential.ipynb`, `Dimer-Simulation.ipynb`).
  - Presentation slides (page 21 to page 37).
  - Project paper: sections relevant to the work above.

## Authors
Copyright (c) 2019
- Wei-Tse Hsu (wehs7661@colorado.edu)
- Lenny Fobe (thfo9888@colorado.edu )