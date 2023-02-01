![header_narrow](https://user-images.githubusercontent.com/26778894/215681761-68adbc1c-4cfa-445d-a745-79a6c09118b2.jpg)

[![License](http://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/polyphy/polyphy/main/LICENSE)
[![Python
Package](https://github.com/PolyPhyHub/PolyPhy/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/PolyPhyHub/PolyPhy/actions/workflows/python-package.yml)
[![image](https://ci.appveyor.com/api/projects/status/ynv14em7nm0tvjso/branch/main?svg=true)](https://ci.appveyor.com/project/PatriceJada/polyphy-uyogg/branch/main)
[![Documentation
Status](https://readthedocs.org/projects/polyphy/badge/?version=latest)](https://polyphy.readthedocs.io/en/latest/?badge=latest)
[![REUSE
status](https://api.reuse.software/badge/git.fsfe.org/reuse/api)](https://api.reuse.software/info/git.fsfe.org/reuse/api)
<!-- [![image](https://codecov.io/gh/PolyPhyHub/PolyPhy/branch/main/graph/badge.svg?token=D933raYfrG)](https://codecov.io/gh/PolyPhyHub/PolyPhy) -->

# PolyPhy
*PolyPhy* is an unconventional toolkit for reconstructing continuous networks out of sparse 2D or 3D data. Such data can be defined as collections of discrete points, or a continuous sparse scalar field. *PolyPhy* produces a scalar density field that defines the recovered network structure. With the help of GPU-accelerated simulation and visualization, *PolyPhy* provides domain experts an interactive way to reconstruct discrete geometric data with an underlying network structure. The reconstruction is driven by the *Monte Carlo Physarum Machine* algorithm, a metaheuristic inspired by the morphology and dynamics of Physarum polycephalum aka 'slime mold'.

## Related Resources
- *PolyPhy* is a successor of [Polyphorm](https://github.com/CreativeCodingLab/Polyphorm)
- Main GitHub page: [PolyPhy Hub](https://github.com/PolyPhyHub)
- Official website: [polyphy.io](https://polyphy.io)
- Underlying research: [overview](https://elek.pub/projects/Rhizome-Cosmology/) and [publications](https://elek.pub/research.html)

## System Requirements
- Decent GPU, currently tested NVIDIA GPUs, other brands subject to support by Taichi
- Recent Windows, Linux or Mac OS
- Python 3.x, Anaconda recommended

## Repository
The repository is located at the following GitHub URL:<br/>
<https://github.com/PolyPhyHub/PolyPhy.git>

## Functionality
The use-cases currently supported by *PolyPhy* are divided according to the data workflow they are built around. Each use-case has a corresponding Jupyter notebook that implements it located in **./experiments/Jupyter**. This section reviews them case by case, and the following section provides an extensive tutorial recorded at the recent OSPO Symposium 2022.

- **2D self-patterning** is the most basic use-case implemented within the **./experiments/Jupyter/PolyPhy_2D_discrete_data** notebook. The ability of MCPM to generate a diversity of patterns with network characteristics is achieved by disabling the data marker deposition, leaving only the MCPM agents to generate the marker responsible for maintaining structure.<p>
![2D_self-patterning](https://user-images.githubusercontent.com/26778894/215976261-d9509124-e3bf-4b82-9cc8-b96a40ab3db2.jpg)
</p>

## How to Use PolyPhy
Below is a recording of the [PolyPhy Workshop](https://elek.pub/workshop_cross2022.html) given as part of the [OSPO Symposium 2022](https://ospo.ucsc.edu/event/20220927/).<br/>
This 93-minute workshop covers *PolyPhy*'s research background, all of the 5 above usecases, and extended technical discussion.

[![](http://i3.ytimg.com/vi/3-hm7iTqz0U/hqdefault.jpg)](https://www.youtube.com/watch?v=3-hm7iTqz0U "PolyPhy Workshop")

## PolyPhy Build

### Install locally from code
Navigate to the repository root and run:<br/>
```
pip install . -U
```

### Install from PyPI
From the command line run:<br/>
```
pip install polyphy
```

### Running PolyPhy
Please note that the core PolyPhy code is currently undergoing refactoring and the latest functionality presented in the Jupyter notebooks described above is not fully ported yet.

Running PolyPhy from the command line:<br/>
``` pycon
✗ polyphy
[Taichi] version 1.0.3, llvm 10.0.0, commit fae94a21, osx, python 3.8.9
usage: polyphy [-h] [-v] [-q] {run2d,run3d} ...

positional arguments:
  {run2d,run3d}  sub command help
    run2d        run 2D PolyPhy
    run3d        run 3D PolyPhy

optional arguments:
  -h, --help     show this help message and exit
  -v, --version  show program's version number and exit
  -q, --quiet    suppress output
```

Running polyphy2d/3d using the Python interface:<br/>
``` pycon
✗ python
Python 3.8.9 (default, Apr 13 2022, 08:48:06)
[Clang 13.1.6 (clang-1316.0.21.2.5)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import polyphy
[Taichi] version 1.0.3, llvm 10.0.0, commit fae94a21, windows, python 3.8.9
>>> polyphy.lib.run_2D()
[Taichi] Starting on arch=CUDA
...
```

## Services

#### Tox
Tox is a virtual environment management and test tool that allows you to define and run custom tasks that call executables from Python packages. Tox will download the dependencies you have specified, build the package, install it in a virtual environment and run the tests using pytest. Make sure to install tox in the root of your project if you intend to work on the development.

``` pycon
tox # download dependencies, build and install package, run tests
tox -e docs  # to build your documentation
tox -e build  # to build your package distribution
tox -e publish  # to test your project uploads correctly in test.pypi.org
tox -e publish --repository pypi  # to release your package to PyPI
tox -av  # to list all the tasks available
```

#### GitHub Actions
GitHub Actions is being used to test on MacOs as well as Linux. It allows for the automation of the building, testing, and deployment pipline.

#### Codecov
A service that generates a visual report of how much code has been tested. All configuration settings can be found in the codecov.yml file.

#### Appveyor
A service that can be used to test Windows. All configuration settings can be found in the appveyor.yml file.
