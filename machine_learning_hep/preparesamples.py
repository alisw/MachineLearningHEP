###############################################################
##                                                           ##
##     Software for single-label classification with Scikit  ##
##      Origin: G.M. Innocenti (CERN)(ginnocen@cern.ch)       ##
##                                                           ##
###############################################################

"""
Methods to load and prepare data for training
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def prep_mlsamples(df_sig, df_bkg, namesig, nevt_sig, nevt_bkg, test_frac, rnd_splt):

    if nevt_sig > len(df_sig):
        print("there are not enough signal events ")
    if nevt_bkg > len(df_bkg):
        print("there are not enough background events ")

    nevt_sig = min(len(df_sig), nevt_sig)
    nevt_bkg = min(len(df_bkg), nevt_bkg)

    print("used number of signal events are %d" % (nevt_sig))
    print("used number of signal events are %d" % (nevt_bkg))

    df_sig = df_sig[:nevt_sig]
    df_bkg = df_bkg[:nevt_bkg]
    df_sig[namesig] = 1
    df_bkg[namesig] = 0
    df_ml = pd.DataFrame()
    df_ml = pd.concat([df_sig, df_bkg])
    df_ml_train, df_ml_test = train_test_split(df_ml, test_size=test_frac, random_state=rnd_splt)

    print("%d events for training and %d for testing" % (len(df_ml_train), len(df_ml_test)))
    return df_ml_train, df_ml_test
