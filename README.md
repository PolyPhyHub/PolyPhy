# Polyphy

[![Documentation Status]][1]
[![Python Package]][2]
[![License]][3]
[![image]][4]
[![image][5]][6]

## Tox

Tox is a virtual environment management and test tool that allows you to
define and run custom tasks that call executables from Python packages.

Make sure to install tox in the root of your project.

    tox # will download the dependencies you have specified, build the package, install it in a virtual environment and run the tests using pytest.

    tox -e docs  # to build your documentation
    tox -e build  # to build your package distribution
    tox -e publish  # to test your project uploads correctly in test.pypi.org
    tox -e publish -- --repository pypi  # to release your package to PyPI
    tox -av  # to list all the tasks available

  [Documentation Status]: https://readthedocs.org/projects/polyphy/badge/?version=latest
  [1]: https://polyphy.readthedocs.io/en/latest/?badge=latest
  [Python Package]: https://github.com/PolyPhyHub/PolyPhy/actions/workflows/python-package.yml/badge.svg?branch=main
  [2]: https://github.com/PolyPhyHub/PolyPhy/actions/workflows/python-package.yml
  [License]: http://img.shields.io/badge/license-MIT-blue.svg
  [3]: https://raw.githubusercontent.com/polyphy/polyphy/main/LICENSE
  [image]: https://codecov.io/gh/PolyPhyHub/PolyPhy/branch/main/graph/badge.svg?token=D933raYfrG
  [4]: https://codecov.io/gh/PolyPhyHub/PolyPhy
  [5]: https://ci.appveyor.com/api/projects/status/ynv14em7nm0tvjso/branch/main?svg=true
  [6]: https://ci.appveyor.com/project/PatriceJada/polyphy-uyogg/branch/main