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

"""
Methods wrapping/handling Bayesian hyper-parameters optimization for all models
"""


def do_bayesian_opt(bayes_optimisers, x_train, y_train, out_dirs, ncores=-1):
    """Do Bayesian optimisation for all registered models
    """
    for opt, out_dir in zip(bayes_optimisers, out_dirs):
        opt.x_train = x_train
        opt.y_train = y_train

        opt.optimise(ncores=ncores)
        opt.save(out_dir)
        opt.plot(out_dir)
