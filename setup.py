#############################################################################
##  © Copyright CERN 2018. All rights not expressly granted are reserved.  ##
##                 Author: Gian.Michele.Innocenti@cern.ch                  ##
## This program is free software: you can redistribute it and/or modify it ##
##  under the terms of the GNU General Public License as published by the  ##
## Free Software Foundation, either version 3 of the License, or (at your  ##
## option) any later version. This program is distributed in the hope that ##
##  it will be useful, but WITHOUT ANY WARRANTY; without even the implied  ##
##     warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    ##
##           See the GNU General Public License for more details.          ##
##    You should have received a copy of the GNU General Public License    ##
##   along with this program. if not, see <https://www.gnu.org/licenses/>. ##
#############################################################################

from setuptools import setup, find_packages
from codecs import open
from os import path
from glob import glob

setup(
  name='MachineLearningHEP',

  # LAST-TAG is a placeholder. Automatically replaced at deploy time with the right tag
  version='v0.0.3',

  description='Machine Learning package for HEP',

  # Long description from Markdown -- https://dustingram.com/articles/2018/03/16/markdown-descriptions-on-pypi
  # Filter out lines that look like GitHub "badges"
  long_description="\n".join([ line for line in open('README.md').read().split("\n") if not line.startswith("[![") ]),
  long_description_content_type='text/markdown',

  url='https://github.com/ginnocen/MachineLearningHEP',
  author='Gian Michele Innocenti',
  author_email='gian.michele.innocenti@cern.ch',
  license='GPL',
  classifiers=[

    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 4 - Beta',

    # Indicate who your project is intended for
    'Intended Audience :: Education',
    'Topic :: Scientific/Engineering :: Physics',

    # Pick your license as you wish (should match "license" above)
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3.6',
    ],

  # What does your project relate to?
  keywords='HEP Computing MachineLearning',

  # You can just specify the packages manually here if your project is
  # simple. Or you can use find_packages().
  packages=find_packages(),

  # Alternatively, if you want to distribute just a my_module.py, uncomment
  # this:
  #   py_modules=["my_module"],

  # List run-time dependencies here.  These will be installed by pip when
  # your project is installed. For an analysis of "install_requires" vs pip's
  # requirements files see:
  # https://packaging.python.org/en/latest/requirements.html
  install_requires=[ "numpy==1.17.4", "pandas==0.24.2", "scipy==1.4.1", "matplotlib==3.0.3",
                     "seaborn==0.11.1", "uproot==3.4.18", "scikit-learn==0.20.3", "xgboost==0.90",
                     "keras==2.3.1", "tensorflow==2.5.2", "PyYaml==5.1", "pylint",
                     "twisted==19.2.0", "klein==17.10.0", "Jinja2==2.10.3", "numba==0.48.0",
                     "pyarrow==0.13.0", "lz4==2.1.10", "hyperopt==0.2.3", "shap==v0.23.0"],

  python_requires='>=3.6, <3.7',

  # List additional groups of dependencies here (e.g. development
  # dependencies). You can install these using the following syntax,
  # for example:
  # $ pip install -e .[dev,test]
  extras_require={
  },

  # If there are data files included in your packages that need to be
  # installed, specify them here. Note that you need to specify those files in
  # MANIFEST.in as well, since Python tools behave inconsistently
  include_package_data=True,
  package_data={ "machine_learning_hep.submission": [ "default_complete.yml",
                                                      "default_ana.yml",
                                                      "default_apply.yml",
                                                      "default_pre.yml",
                                                      "default_train.yml" ],
                 "machine_learning_hep.data": [ "config_model_parameters.yml",
                                                "database_run_list.yml" ] },

  # Although 'package_data' is the preferred approach, in some case you may
  # need to place data files outside of your packages. See:
  # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
  # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
  data_files=[],

  # To provide executable scripts, use entry points in preference to the
  # "scripts" keyword. Entry points provide cross-platform support and allow
  # pip to create the appropriate form of executable for the target platform.
  # See: https://chriswarrick.com/blog/2014/09/15/python-apps-the-right-way-entry_points-and-scripts/
  entry_points={
      "console_scripts": [ "ml-get-data = machine_learning_hep.ml_get_data:main",
                           "ml-analysis = " \
                           "machine_learning_hep.steer_analysis:main" ]
  }
)
