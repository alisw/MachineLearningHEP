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
Methods to load and prepare data for training
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from machine_learning_hep.logger import get_logger


def prep_mlsamples(df_sig, df_bkg, namesig, nevt_sig, nevt_bkg, test_frac, rnd_splt):

    logger = get_logger()
    if nevt_sig > len(df_sig):
        logger.warning("There are not enough signal events")
    if nevt_bkg > len(df_bkg):
        logger.warning("There are not enough background events")

    nevt_sig = min(len(df_sig), nevt_sig)
    nevt_bkg = min(len(df_bkg), nevt_bkg)

    logger.info("Used number of signal events is %d", nevt_sig)
    logger.info("Used number of background events is %d", nevt_bkg)

    df_sig = df_sig[:nevt_sig]
    df_bkg = df_bkg[:nevt_bkg]
    df_sig[namesig] = 1
    df_bkg[namesig] = 0
    df_ml = pd.DataFrame()
    df_ml = pd.concat([df_sig, df_bkg])
    df_ml_train, df_ml_test = train_test_split(df_ml, test_size=test_frac, random_state=rnd_splt)

    logger.info("%d events for training and %d for testing", len(df_ml_train), len(df_ml_test))
    return df_ml_train, df_ml_test
