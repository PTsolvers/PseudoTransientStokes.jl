# PseudoTransientStokes.jl

Parallel (multi-) XPU iterative 2D and 3D incompressible Stokes flow solvers with viscous and Maxwell visco-elastic shear rheology. This software is part of the [the PTsolvers project](https://ptsolvers.github.io/).

The aim of [the PTsolvers project](https://ptsolvers.github.io/) is to examplify, test and asses the performance of the pseudo-transient method, implementing second-order convergence acceleration building upon the second order Richardson method \[[Frankel, 1950](https://doi.org/10.2307/2002770)\].


> 💡 Link to the [Overleaf draft](https://www.overleaf.com/project/5ff83a57858b372f63143b8e)

## Content
* [Stokes flow](#stokes-flow)
* [Scripts](#scripts)
* [Additional infos](#additional-infos)


## Stokes flow


## Scripts

### Optimal iteration parameters
The folder [**dispersion_analysis**](/dispersion_analysis) contains the analytical derivations for the values of iteration parameters. We provide these derivations for 1D viscous Stokes problem. Only the case of `μ=const` is considered.

The main output of the script is the theoretically predicted value for the non-dimensional parameters `Re` and `r`, which are used in the solvers. The figure showing the dependency of the residual decay rate on `Re` and `r` is also displayed:

<img src="dispersion_analysis/fig_dispersion_analysis_stokes1D.png" alt="Results of the dispersion analysis for the stokes problem" width="500">

For users' convenience, we provide two versions of each script, one version written in Matlab and the other in Python.

To launch the Matlab version, the working installation of Matlab and [Matlab Symbolic Math Toolbox](https://www.mathworks.com/products/symbolic.html) is required.

The second version is implemented using the open-source computer algebra library [SymPy](https://www.sympy.org/) as a [Jupyter](https://jupyter.org/)/[IPython](https://ipython.org/) notebook. The Jupyter notebooks can be viewed directly at GitHub ([example](https://github.com/PTsolvers/PseudoTransientStokes.jl/blob/main/dispersion_analysis/dispersion_analysis_stokes1D.ipynb)). However, in order to view the notebook on a local computer or to make changes to the scripts, the recent Python installation is required. Also, several Python packages need to be installed: SymPy, NumPy, Jupyter, and Matplotlib. The easiest way to install these packages along with their dependencies is to use the [Anaconda](https://www.anaconda.com/products/individual) platform.

After installing `Anaconda`, open the terminal, `cd` into the `dispersion_analysis` folder and create a new `conda` environment with the following command:
```
> conda create -n ptsolvers sympy numpy jupyter matplotlib
```
This command will install the required packages. After the installation completes, activate the environment:
```
> conda activate ptsolvers
```
The final step is to launch the Jupyter server:
```
> jupyter notebook
```
This command starts a server and opens the browser window with file manager.


## Additional infos
