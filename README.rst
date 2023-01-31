=======
Polyphy
=======

.. image:: https://readthedocs.org/projects/polyphy/badge/?version=latest
   :target: https://polyphy.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://github.com/PolyPhyHub/PolyPhy/actions/workflows/python-package.yml/badge.svg?branch=main
   :target: https://github.com/PolyPhyHub/PolyPhy/actions/workflows/python-package.yml
   :alt: Python Package
.. image:: http://img.shields.io/badge/license-MIT-blue.svg
   :target: https://raw.githubusercontent.com/polyphy/polyphy/main/LICENSE
   :alt: License
.. image:: https://codecov.io/gh/PolyPhyHub/PolyPhy/branch/main/graph/badge.svg?token=D933raYfrG
   :target: https://codecov.io/gh/PolyPhyHub/PolyPhy
.. image:: https://ci.appveyor.com/api/projects/status/ynv14em7nm0tvjso/branch/main?svg=true
   :target: https://ci.appveyor.com/project/PatriceJada/polyphy-uyogg/branch/main
.. image:: https://api.reuse.software/badge/git.fsfe.org/reuse/api
   :target: https://api.reuse.software/info/git.fsfe.org/reuse/api
   :alt: REUSE status
   

How to use polyphy
==================

Installation
------------

Install from the source.

1. Clone the repo: https://github.com/PolyPhyHub/PolyPhy.git
2. Go to the directory and run **pip install . -U**

Install from pypi.

1. **pip install polyphy**

Running polyphy
---------------

Using command line

.. code-block:: pycon

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

Run polyphy2d using python interface

.. code-block:: pycon

    ✗ python
    Python 3.8.9 (default, Apr 13 2022, 08:48:06)
    [Clang 13.1.6 (clang-1316.0.21.2.5)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import polyphy
    [Taichi] version 1.0.3, llvm 10.0.0, commit fae94a21, osx, python 3.8.9
    >>> polyphy.lib.run_2D()
    [Taichi] Starting on arch=metal

Tox
===

Tox is a virtual environment management and test tool that allows you to define and run custom tasks that call executables from Python packages.

Make sure to install tox in the root of your project.

.. code-block:: pycon

    tox # will download the dependencies you have specified, build the package, install it in a virtual environment and run the tests using pytest.
    tox -e docs  # to build your documentation
    tox -e build  # to build your package distribution
    tox -e publish  # to test your project uploads correctly in test.pypi.org
    tox -e publish -- --repository pypi  # to release your package to PyPI
    tox -av  # to list all the tasks available

Services
========

Codecov
--------

A service that generates a visual report of how much code has been tested. All configuration settings can be found in the codecov.yml file.

GitHub Actions
--------------

GitHub Actions is being used to test on MacOs as well as Linux. It allows for the automation of the building, testing, and deployment pipline.

Appveyor
--------

A service that can be used to test Windows. All configuration settings can be found in the appveyor.yml file.


