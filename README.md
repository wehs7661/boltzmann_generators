Boltzmann Generators
==============================

## Description
This is a repository for the final project of the course Computational Statistical Physics (PHYS 7810) at CU Boulder, of which the goal is to develop and apply [Boltzmann generators](https://science.sciencemag.org/content/365/6457/eaaw1147) to different molecular systems. The project paper can be found in the folder `Project`. Specifically, this repository includes the following directories:
- `Project`: A folder containing the project paper and the presentation slides for this final proejct.
- `References`: A folder containing the most important journal papers relevant to our project.
- `Notebooks`: A folder containing severl jupyter notebooks in a tutorial style which implements Boltzmann generators an applies them to different systems of interest. From the introductory tutorial of PyTorch and the simplest toy model, to a more advanced molecular system, these notebooks include:
  - `PyTorch Introduction.ipynb`: A jupyter notebook adapted from a tutorial generously provided by the course [CSE446: Machine Learning at Univeristy of Washington](https://courses.cs.washington.edu/courses/cse446/19au/section9.html).
  - `Double-well Potential.ipynb`: A notebook which implements the architecture of Boltzmann generators and applies them to the simplest toy model: double-well potential.
  - `Mueller Brown Potential.ipyng`: A notebook which applies Boltzmann generators to the slightly more complex Mueller potential, which is characteristic of three energy minima and a more complicated reaction coordinate.
  - `Dimer-Simulation.ipynb`: A notebook which applies Boltzmann generators to a dimer in Lennard-Jones bath.
  - `BPTI protein.ipynb`: A notebook which applies Boltzmann generators to a real biomolecular system to explore different confomrations of the BPTI protein.
- `Library`: A folder of software implementing Boltzmann generators that can be imported by the jupyter notebooks in the folder `Notebooks`.


## Authors

Copyright (c) 2019
- Lenny Fobe (thfo9888@colorado.edu )
- Wei-Tse Hsu (wehs7661@colorado.edu)