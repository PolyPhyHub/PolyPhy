![header_narrow](https://user-images.githubusercontent.com/26778894/215681761-68adbc1c-4cfa-445d-a745-79a6c09118b2.jpg)

[![PolyPhyHub - PolyPhy](https://img.shields.io/static/v1?label=PolyPhyHub&message=PolyPhy&color=blue&logo=github)](https://github.com/PolyPhyHub/PolyPhy "Go to GitHub repo")
[![License](http://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/PolyPhyHub/PolyPhy/main/LICENSE)
[![issues - PolyPhy](https://img.shields.io/github/issues/PolyPhyHub/PolyPhy)](https://github.com/PolyPhyHub/PolyPhy/issues)
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
- Email list: [Google Groups](https://groups.google.com/g/polyphy-news)

## System Requirements
- Decent GPU, recommended a mid-range discrete one
  - currently running best on NVIDIA GPUs, other brands supported as well (subject to the current capabilities of the [Taichi API](https://github.com/taichi-dev/taichi))
  - CPU fallback available for debugging purposes
- Recent Windows, Linux or Mac OS
- Python 3.x, Anaconda recommended

## Repository
The main repository is located at the following GitHub URL:<br/>
<https://github.com/PolyPhyHub/PolyPhy.git>

The other repositories are linked from the following "org" page:<br/>
<https://github.com/PolyPhyHub/>

## Running PolyPhy
Please note the project is currently undergoing a significant refactoring in order to streamline the use of the software in CLI, improve its modularity (making it easier to implement custom pipelines and extend the existing ones), and add new features (such as the recent addition of batch mode).

To **install** PolyPhy, clone this repository, open a Python console, navigate to the root of the repo, and run
```
pip install -r requirements.txt
```
Afterwards, navigate to the **./experiments/jupyter/production/** and run
```
python polyphy_2DDiscrete.py -f "data/csv/sample_2D_linW.csv"
```
for the standard 2D pipeline, or
```
python polyphy_3DDiscrete.py -f "data/csv/sample_3D_linW.csv"
```
to invoke the standard 3D discrete pipeline on sample data. You can also specify a custom CSV file (see the sample data for the format details, typically the data are tuples with 2 or 3 spatial coorinates followed by weights for each data point). The functionality of these pipelines is described below.

To display help on the available CLI parameters, simply run the respective command without any arguments.

There is also a number of notebooks implementing various pipelines (some of which are documented below). These are updated to different degrees, and we are in the process of porting them to the refactored class structure. Updates coming soon.

## Functionality
The use-cases currently supported by *PolyPhy* are divided according to the data workflow they are built around. Each use-case has a corresponding Jupyter notebook that implements it located in **./experiments/Jupyter**. This section reviews them case by case, and the following section provides an extensive tutorial recorded at the recent OSPO Symposium 2022.

- **2D self-patterning** is the most basic use-case implemented within the **./experiments/Jupyter/PolyPhy_2D_discrete_data** notebook. The ability of MCPM to generate a diversity of patterns with network characteristics is achieved by disabling the data marker deposition, leaving only the MCPM agents to generate the marker responsible for maintaining structure.<p>
  ![2D_self-patterning](https://user-images.githubusercontent.com/26778894/215976261-d9509124-e3bf-4b82-9cc8-b96a40ab3db2.jpg)
</p>

- **2D procedural pipeline** provide an easy environment to experiment with the behavior of *PolyPhy* in the presence of discrete data with different spatial frequencies. Editing (adding new data points) is also supported. This pipeline is implemented in the **./experiments/Jupyter/PolyPhy_2D_discrete_data** notebook.<p>
  ![2D_discrete_procedural](https://user-images.githubusercontent.com/26778894/215980005-f927d227-0090-46dd-8ec6-fde9b800dfa0.jpg)
</p>

- **2D discrete pipeline** implements the canonical way of working with custom data defined by a CSV file. The example below demonstrates fitting to a 2D projection of the SDSS galaxy dataset. This pipeline is implemented in the **./experiments/Jupyter/PolyPhy_2D_discrete_data** notebook.<p>
  ![2D_discrete_explicit](https://user-images.githubusercontent.com/26778894/215980486-f77da2ec-8780-4a23-bacc-a03c164ebe2a.jpg)
</p>

- **2D continuous pipeline** demonstrates the workflow with a continuous user-provided dataset. Instead of a discrete set of points as in the previous use-cases, the data is defined by a scalar field, which in 2D amounts to a grayscale image. The example below approximates the US road network using only a sparse population density map as the input. This pipeline is implemented in the **./experiments/Jupyter/PolyPhy_2D_continuous_data** notebook.<p>
  ![2D_continuous](https://user-images.githubusercontent.com/26778894/215981222-6fa4b334-45d2-498f-8c5a-c150137574ac.jpg)
</p>

- **3D discrete pipeline** represents an equivalent functionality to the original *Polyphorm* implementation. The dataset consists of SDSS galaxies defined as a weighted collection of 3D points. THe visualization is based on volumetric ray marching simultaneously fetching the deposit and the trace fields. This pipeline is implemented in the **./experiments/Jupyter/PolyPhy_3D_discrete_data** notebook.<p>
  ![3D_discrete_explicit](https://user-images.githubusercontent.com/26778894/215981925-96ed3322-0068-497d-a2e7-4543c7ef8e41.jpg)
</p>

## How to Use PolyPhy
Below is a recording of the [PolyPhy Workshop](https://elek.pub/workshop_cross2022.html) given as part of the [OSPO Symposium 2022](https://ospo.ucsc.edu/event/20220927/).<br/>
This 93-minute workshop covers *PolyPhy*'s research background, all of the 5 above usecases, and extended technical discussion.

[![](http://i3.ytimg.com/vi/3-hm7iTqz0U/hqdefault.jpg)](https://www.youtube.com/watch?v=3-hm7iTqz0U "PolyPhy Workshop")

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
