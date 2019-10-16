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
    #return XGBClassifier(n_gpus=0, n_jobs=10,
    #                    objective=model_config['objective'],
    #                     min_child_weight=model_config['min_child_weight'],
    #                     max_depth=model_config['max_depth'],
    #                     gamma=model_config['gamma'],
    #                     subsample=model_config['subsample'],
    #                     colsample_bytree=model_config['colsample_bytree'],
    #                     learning_rate=model_config['learning_rate'],
    #                     n_estimators=model_config['n_estimators'],
    #			 reg_alpha=model_config['reg_alpha'],
    #			 scale_pos_weights = model_config['scale_pos_weights'])
    return XGBClassifier(n_gpus=0, n_jobs=10, tree_method="hist",
                         min_child_weight=model_config['min_child_weight'],
                         max_depth=model_config['max_depth'],
                         gamma=model_config['gamma'],
                         subsample=model_config['subsample'],
                         colsample_bytree=model_config['colsample_bytree'],
                         learning_rate=model_config['learning_rate'],
                         n_estimators=model_config['n_estimators'])
   #return XGBClassifier(n_gpus=0, nthread=-1, objective=model_config['objective'], max_depth=model_config['max_depth'], subsample=model_config['subsample'],
   #colsample_bytree=model_config['colsample_bytree'], booster=model_config['booster'])



