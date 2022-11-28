# Classical Simulation of Dipolar Atoms in a Trap
A brute force optimisation method for finding local minimums of the energy of dipolar atoms in a quadratic trap and the configurations of atomic postitions at this energy.

## Outline
- `Energy_Derivative`: File containing functions for calculating the energy of a given configuration of atomic cartesian positions and to produce an array of derivatives with respect to each coordinate of the energy function for the given configuration of atomic positions.
- `Optimisation_Code`: File that runs the optisation method "bfgs" for finding local minimums for a given number of iterations.
- `Plotting`: Script that reads data produced from main to produce a number of plots.

## Libraries Used

- Python libraries:
    - NumPy
    - matplotlib
    - scipy
    - pandas
