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
Metrics for (ML) optimisation
"""
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score


def get_scorers(score_names):
    """Construct dictionary of scorers

    Args:
        score_names: tuple of names. Available names see below
    Returns:
        dictionary mapping scorers to names
    """

    scorers = {}
    for sn in score_names:
        if sn == "AUC":
            scorers["AUC"] = make_scorer(roc_auc_score, needs_threshold=True)
        elif sn == "Accuracy":
            scorers["Accuracy"] = make_scorer(accuracy_score)

    return scorers
