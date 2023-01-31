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
*PolyPhy* is an unconventional toolkit for reconstructing continuous networks out of sparse 2D or 3D data. Such data can be defined as collections of discrete points, or a continuous sparse scalar field. *PolyPhy* produces a scalar density field that defines the recovered network structure. With the help of GPU-accelerated simulation and visualization, *PolyPhy* provides domain experts an interactive way to reconstruct discrete geometric data with an underlying network structure.

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
The repository is located on the following GitHub URL:<br/>
<https://github.com/PolyPhyHub/PolyPhy.git>

## How to Use PolyPhy
Below is a recording of the [PolyPhy Workshop](https://elek.pub/workshop_cross2022.html) given as part of the [OSPO Symposium 2022](https://ospo.ucsc.edu/event/20220927/).<br/>
This 93-minute workshop covers *PolyPhy*'s research background, all 4 above usecases, and technical discussion.

[![](http://i3.ytimg.com/vi/3-hm7iTqz0U/hqdefault.jpg)](https://www.youtube.com/watch?v=3-hm7iTqz0U "PolyPhy Workshop")

## Functionality
The 3 main use cases detailed in the above workshop are currently implemented in dedicated Jupyter notebooks located in **./experiments/Jupyter**. They are as follows, in regard to the data pipeline they implement: discrete 2D, continuous 2D data and discrete 3D data.

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
