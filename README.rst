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

Tox
===

Tox is a virtual environment management and test tool that allows you to define and run custom tasks that call executables from Python packages.

Make sure to install tox in the root of your project. 

::

    tox # will download the dependencies you have specified, build the package, install it in a virtual environment and run the tests using pytest.
::

    tox -e docs  # to build your documentation
    tox -e build  # to build your package distribution
    tox -e publish  # to test your project uploads correctly in test.pypi.org
    tox -e publish -- --repository pypi  # to release your package to PyPI
    tox -av  # to list all the tasks available
