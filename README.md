<p align="center">
  <img src="https://github.com/PolyPhyHub/PolyPhy/blob/main/third_party/Polyphy-logo.png">
</p>

<p align="center">
  <em>PolyPhy reconstructs continuous networks from sparse 2D or 3D data using GPU-accelerated simulation with the MCPM algorithm.</em>

<p align="center">
<a href="https://github.com/PolyPhyHub/PolyPhy" title="Go to GitHub repo"><img src="https://img.shields.io/static/v1?label=PolyPhyHub&message=PolyPhy&color=blue&logo=github" /></a>
<a href="https://raw.githubusercontent.com/PolyPhyHub/PolyPhy/main/LICENSE"><img src="http://img.shields.io/badge/license-MIT-blue.svg" /></a>
<a href="https://github.com/PolyPhyHub/PolyPhy/issues"><img src="https://img.shields.io/github/issues/PolyPhyHub/PolyPhy" /></a>
<a href="https://github.com/PolyPhyHub/PolyPhy/actions/workflows/python-package.yml"><img src="https://github.com/PolyPhyHub/PolyPhy/actions/workflows/python-package.yml/badge.svg?branch=main" /></a>
<a href="https://ci.appveyor.com/api/projects/status/32r7s2skrgm9ubva?svg=true"><img src="https://ci.appveyor.com/api/projects/status/32r7s2skrgm9ubva?svg=true" /></a>
<a href="https://polyphy.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/polyphy/badge/?version=latest" /></a>
<a href="https://api.reuse.software/info/git.fsfe.org/reuse/api"><img src="https://api.reuse.software/badge/git.fsfe.org/reuse/api" /></a>
<!-- <a href="https://codecov.io/gh/PolyPhyHub/PolyPhy"><img src="https://codecov.io/gh/PolyPhyHub/PolyPhy/branch/main/graph/badge.svg?token=D933raYfrG" /></a> -->
</p>


# PolyPhy
*PolyPhy* is an unconventional toolkit for reconstructing continuous networks out of sparse 2D or 3D data. Such data can be defined as collections of discrete points, or a continuous sparse scalar field. *PolyPhy* produces a scalar density field that defines the recovered network structure. With the help of GPU-accelerated simulation and visualization, *PolyPhy* provides domain experts an interactive way to reconstruct discrete geometric data with an underlying network structure. The reconstruction is driven by the *Monte Carlo Physarum Machine* algorithm, a metaheuristic inspired by the morphology and dynamics of Physarum polycephalum aka 'slime mold'. 

## Related Resources
- *PolyPhy* is a successor of [Polyphorm](https://github.com/CreativeCodingLab/Polyphorm)
- Main GitHub page: [PolyPhy Hub](https://github.com/PolyPhyHub)
- Official website: [polyphy.io](https://polyphy.io)
- Underlying research: [overview](https://elek.pub/projects/Rhizome-Cosmology/) and [publications](https://elek.pub/research.html)
- Email list: [Google Groups](https://groups.google.com/g/polyphy-news)

## System Requirements
- Decent GPU, recommended a mid-range discrete NVIDIA/AMD device
  - currently running best on NVIDIA GPUs, other vendors supported as well (subject to the current capabilities of the [Taichi API](https://github.com/taichi-dev/taichi))
  - CPU fallback available for debugging purposes
- Corresponding GPU environment and drivers (e.g. Vulkan, Metal)
- Recent Windows, Linux or Mac OS
- Python 3.x, Anaconda recommended

## Repository
The main repository is located at the following GitHub URL: <br/>
<https://github.com/PolyPhyHub/PolyPhy.git>

The other repositories are linked from the following "org" page: <br/>
<https://github.com/PolyPhyHub/>

## Running PolyPhy
Please note the project is currently undergoing refactoring in order to streamline the use of the software in CLI, improve its modularity (making it easier to implement custom pipelines and extend the existing ones), and add new features (such as the recent addition of batch mode).

To **install** PolyPhy, clone this repository, open a Python console, navigate to the root of the repo, and run
```
pip install -r requirements.txt
```
Afterwards, navigate to **./src/polyphy** and run
```
python polyphy.py 2d_discrete -f "data/csv/sample_2D_linW.csv" -t 1440
```
for the standard 2D pipeline using the provided sample 2D dataset and max trace resolution of 1440 (which will also determine the window resolution), or
```
python polyphy.py 3d_discrete -f "data/csv/sample_3D_linW.csv -t 200 -x 0 -y 0"
```

to invoke the standard 3D discrete pipeline on sample data, with max trace resolution of 200. You can also specify a custom CSV file (see the sample data for the format details, typically the data are tuples with 2 or 3 spatial coorinates followed by weights for each data point). The functionality of these pipelines is described below.

- `-x`: The -x flag is used to specify a value related to the X-axis.
- `-y`: The -y flag is used to specify a value related to the Y-axis.
- `-t`: The -t flag usually stands for "time" or "iterations"

To display help on the available CLI parameters, simply run the respective command without any arguments.

There is also a number of Jupyter notebooks implementing various experiemtal pipelines (some of which are documented below). These are updated to different degrees, and we are in the process of porting them to the refactored class structure. Updates coming soon.

## Functionality
The use-cases currently supported by *PolyPhy* are divided according to the data workflow they are built around. Each use case has a corresponding (extensible) pipeline specified as a command line parameter under its name. Experimental pipelines not yet implemented in the main build are located in **./experiments/Jupyter**. This section reviews them case by case, and the following section provides an extensive tutorial recorded at the recent OSPO Symposium 2022.

- **2D self-patterning** is the most basic use-case implemented within the **2d_discrete** pipeline. The ability of MCPM to generate a diversity of patterns with network characteristics is achieved by *disabling the data marker deposition*, leaving only the MCPM agents to generate the marker responsible for maintaining structure.<p>
  ![2D_self-patterning](https://user-images.githubusercontent.com/26778894/215976261-d9509124-e3bf-4b82-9cc8-b96a40ab3db2.jpg)
</p>

- **2D procedural pipeline** provide an easy environment to experiment with the behavior of *PolyPhy* in the presence of discrete data with different spatial frequencies. Editing (adding new data points) is also supported. This is invoked by specifying **2d_discrete** pipeline without providing any input data file, thus prompting *PolyPhy* to generate the data procedurally.<p>
  ![2D_discrete_procedural](https://user-images.githubusercontent.com/26778894/215980005-f927d227-0090-46dd-8ec6-fde9b800dfa0.jpg)
</p>

- **2D discrete pipeline** implements the canonical way of working with custom data defined by a CSV file. The example below demonstrates fitting to a 2D projection of the SDSS galaxy dataset. It is invoked by specifying **2d_discrete** pipeline and a custom input data file.<p>
  ![2D_discrete_explicit](https://user-images.githubusercontent.com/26778894/215980486-f77da2ec-8780-4a23-bacc-a03c164ebe2a.jpg)
</p>

- **2D continuous pipeline** demonstrates the workflow with a continuous user-provided dataset. Instead of a discrete set of points as in the previous use-cases, the data is defined by a scalar field, which in 2D amounts to a grayscale image. The example below approximates the US road network using only a sparse population density map as the input. This pipeline is implemented in the **./experiments/Jupyter/PolyPhy_2D_continuous_data** notebook, to be ported to the main build.<p>
  ![2D_continuous](https://user-images.githubusercontent.com/26778894/215981222-6fa4b334-45d2-498f-8c5a-c150137574ac.jpg)
</p>

- **3D discrete pipeline** represents an equivalent functionality to the original *Polyphorm* implementation. The dataset consists of SDSS galaxies defined as a weighted collection of 3D points. The visualization is based on volumetric ray marching simultaneously fetching the deposit and the trace fields. This pipeline is invoked through the **3d_discrete** parameter.<p>
  ![3D_discrete_explicit](https://user-images.githubusercontent.com/26778894/215981925-96ed3322-0068-497d-a2e7-4543c7ef8e41.jpg)
</p>

## Hyperparameters used

1. **`Sensing dist`:** Average distance in world units at which agents probe the deposit.  
2. **`Sensing angle`:** Angle in radians within which agents probe deposit (left and right concentric to movement direction).  
3. **`Sampling expo`:** Sampling sharpness or 'acuteness' or 'temperature' which tunes the directional mutation behavior.  
4. **`Step size`:** Average size of the step in world units which agents make in each iteration.  
5. **`Data deposit`:** Amount of marker 'deposit' that *data* emit at every iteration.  
6. **`Agent deposit`:** Amount of marker 'deposit' that *agents* emit at every iteration.  
7. **`Deposit attn`:** Attenuation or 'decay' rate of the diffusing combined agent+data deposit field.  
8. **`Trace attn`:** Attenuation or 'decay' of the non-diffusing agent trace field.  
9. **`Deposit vis`:** Visualization intensity of the green deposit field (logarithmic).  
10. **`Trace vis`:** Visualization intensity of the red trace field (logarithmic).

## Options
1. **`Distance distribution`:** strategy for sampling the sensing and movement distances
2. **`Directional distribution`:** strategy for sampling the sensing and movement directions
3. **`Directional mutation`:** strategy for selecting the new movement direction
4. **`Deposit fetching`:** access behavior when sampling the deposit field
5. **`Agent boundary handling`:** what do agents do if they reach the boundary of the simulation domain

## How to Use PolyPhy
Below is a recording of the [PolyPhy Workshop](https://elek.pub/workshop_cross2022.html) given as part of the [OSPO Symposium 2022](https://ospo.ucsc.edu/event/20220927/).<br/>
This 90-minute workshop covers *PolyPhy*'s research background, all of the 5 above usecases, and extended technical discussion.

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

GitHub Actions used for Polyphy are:

- **`batch_mode.yml`**: Automates testing for the Polyphy project by running `polyphy.py` scripts on Ubuntu, ensuring expected output files are generated.
- **`macos_python.yml`**: Runs tests on MacOS with multiple Python versions, including linting with `flake8` and testing with `pytest`.
- **`python-package.yml`**: Builds and tests the Polyphy Python package on Ubuntu 22.04 across various Python versions, including linting and code coverage reporting.
- **`release_package.yml`**: Publishes the Polyphy package to `PyPI` if a new version is detected, using `Twine` for the upload process.
- **`ubuntu20_python.yml`**: Runs tests on Ubuntu 20.04 with multiple Python versions, including linting, `pytest` for Python files, and `pytest` for Jupyter notebooks.

#### Codecov
A service that generates a visual report of how much code has been tested. All configuration settings can be found in the codecov.yml file.

#### Appveyor
A service that can be used to test Windows. All configuration settings can be found in the appveyor.yml file.

## How to Contribute

We welcome and appreciate new contributions. Please take a moment to review our [Contribution Guidelines](./CONTRIBUTING.rst) to get started.
