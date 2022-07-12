=================
Developer's guide
=================

1. Quickstart by forking the main repository https://github.com/PolyPhyHub/PolyPhy
2. Clone your copy of the repository

    - Using https ``git clone https://github.com/[YourUsername]/polyphy.git``
    - Using ssh ``git clone git@github.com:[YourUsername]/polyphy.git``

3. Link or point your cloned copy to the main repository. (I always name it upstream)

    - ``git remote add upstream https://github.com/PolyPhyHub/PolyPhy.git``

5. Check/confirm your settings using ``git remote -v``

::

    origin	git@github.com:[YourUsername]/polyphy.git (fetch)
    origin	git@github.com:[YourUsername]/polyphy.git (push)
    upstream	https://github.com/PolyPhyHub/PolyPhy.git (fetch)
    upstream	https://github.com/PolyPhyHub/PolyPhy.git (push)

6. Install the package from the main directory.
use `-U or --upgrade` to upgrade or overwrite any previously installed versions.

::

    pip install . -U

7. Check if the package was installed

::

    polyphy -v

Required Modules
================

You will need Python 3.8+
Make sure the required modules are installed: ``Pip install -r requirements.txt``

Developers need to install these extra packages.

::

   pip install codecov
   pip install coverage
   pip install flake8
   pip install matplotlib
   pip install numpy
   pip install pylint
   pip install pytest
   pip install pytest-cov
   pip install pytest-xdist
   pip install setuptools
   pip install sphinx
   pip install taichi
   pip install toml
   pip install tox
   pip install yapf

Installation
============

Either install from source as

.. code-block:: bash

  pip install . --upgrade
  or
  pip install . -U
  or
  python setup.py install

or install in development mode.

.. code-block:: bash

  python setup.py develop
  or
  pip install -e .


For more about `installing`_ refer to the python setuptools `documentation`_.

you can also install from Git.

.. code-block:: bash

  # Local repository
  pip install git+file:///path/to/your/git/repo #  test a PIP package located in a local git repository
  pip install git+file:///path/to/your/git/repo@branch  # checkout a specific branch by adding @branch_name at the end

  # Remote GitHub repository
  pip install git+git://github.com/myuser/myproject  #  package from a GitHub repository
  pip install git+git://github.com/myuser/myproject@my_branch # github repository Specific branch


Running tests locally
^^^^^^^^^^^^^^^^^^^^^

From the source top-level directory, Use Pytest as examples below

.. code-block:: sh

  $   pytest -v # All tests

Using tox

.. code-block:: sh

  $   tox # will download the dependencies you have specified, build the package, install it in a virtual environment and run the tests using pytest.


Style Guide for Python Code
---------------------------

Use ``yapf -d --recursive polyphy/ --style=.style.yapf`` to check style.

Use ``yapf -i --recursive polyphy/ --style=.style.yapf`` refactor style

Continuous Integration
^^^^^^^^^^^^^^^^^^^^^^

The main GitHub repository runs the test on GitHub Actions (Ubuntu-latest and Ubuntu 20.04).

Pull requests submitted to the repository will automatically be tested using
these systems and results reported in the ``checks`` section of the pull request
page.


Create Release
==============

Start
^^^^^

1. **Run the tests**.
2. Update ``CHANGELOG.rst`` with major updates since the last release
3. Update the version number `bumpversion release` or provide a version as `bumpversion --new-version 3.1.0`
4. On Github draft a release with the version changes. Provide a version as tag and publish.
5. After the release, update the version to dev, run `bumpversion patch`

Release on Test PyPi and PyPi is handled by Github actions using tox.

::

    git push upstream main
    git push upstream --tags

Documentation
=============

We are using `Sphinx`_ and `Read the Docs`_. for the documentation.
For documentation refer to the `README under the tox section`_


Collaborative Workflows with GitHub
===================================

This will update your `.git/config` to point to your repository copy of the Polyphy as `remote "origin"`
To fetch pull requests you can add `fetch = +refs/pull/*/head:refs/remotes/origin/pr/*` to your `.git/config` as below.
.. code-block:: bash

  [remote "upstream"]
  url = https://github.com/PolyPhyHub/PolyPhy.git
  fetch = +refs/heads/*:refs/remotes/upstream/*
  # To fetch pull requests add
  fetch = +refs/pull/*/head:refs/remotes/origin/pr/*

Fetch upstream main and create a branch to add the contributions to.

.. code-block:: bash

  git fetch upstream
  git checkout main
  git reset --hard upstream/main
  git checkout -b [new-branch-to-fix-issue]

Please read the `contribution`_ docs.

.. _AppVeyor: https://www.appveyor.com/
.. _codecov: https://codecov.io/
.. _documentation: https://pythonhosted.org/an_example_pypi_project/setuptools.html
.. _installing: https://docs.python.org/3/install/index.html
.. _Read The Docs: https://readthedocs.org/
.. _setup tools: https://pythonhosted.org/an_example_pypi_project/setuptools.html
.. _README under the tox section: readme.html#Tox
.. _contribution: contributing.html
.. _Sphinx: http://www.sphinx-doc.org/en/stable/
