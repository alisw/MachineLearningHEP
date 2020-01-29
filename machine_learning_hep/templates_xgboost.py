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

from xgboost import XGBClassifier

def xgboost_classifier(model_config): # pylint: disable=W0613
    return XGBClassifier(n_gpus=0,
                         n_jobs=model_config['n_jobs'],
                         tree_method=model_config['tree_method'],
                         max_depth=model_config['max_depth'],
                         learning_rate=model_config['learning_rate'],
                         n_estimators=model_config['n_estimators'],
                         objective=model_config['objective'],
                         gamma=model_config['gamma'],
                         min_child_weight=model_config['min_child_weight'],
                         #early_stopping_rounds=model_config['early_stopping_rounds'],
                         subsample=model_config['subsample'],
                         colsample_bytree=model_config['colsample_bytree'],
                         colsample_bynode=model_config['colsample_bynode'],
                         random_state=model_config['random_state']
                         )