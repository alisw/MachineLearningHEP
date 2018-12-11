###############################################################
##                                                           ##
##     Software for single-label classification with Scikit  ##
##      Origin: G.M. Innocenti (CERN)(ginnocen@cern.ch)       ##
##                                                           ##
###############################################################

"""
main macro for running the study
"""
from sklearn.utils import shuffle
from machine_learning_hep.general import get_database_ml_parameters
from machine_learning_hep.general import getdataframe_datamc, filterdataframe_datamc
from machine_learning_hep.preparesamples import preparemlsample


def test():
    data = get_database_ml_parameters()
    nevents = 1000
    mltype = "BinaryClassification"
    mlsubtype = "HFmeson"
    optionanalysis = "Lc"
    var_skimming = ["pt_cand_ML"]
    varmin = [2]
    varmax = [4]
    print(nevents, mltype, mlsubtype, optionanalysis)
    mylistvariables = data[optionanalysis]["var_training"]
    mylistvariablesothers = data[optionanalysis]["var_others"]
    myvariablessignal = data[optionanalysis]["var_signal"]
    myvariablesy = data[optionanalysis]["var_target"]
    mylistvariablesx, mylistvariablesy = data[optionanalysis]["var_correlation"]
    mylistvariablesall = data[optionanalysis]["var_all"]

    print(mylistvariables)
    print(mylistvariablesothers)
    print(myvariablessignal)
    print(myvariablesy)
    print(mylistvariablesx)
    print(mylistvariablesy)
    print(mylistvariablesall)

    loadsampleoption = 1

    if loadsampleoption == 1:
        filedata, filemc = data[optionanalysis]["data_mc_files"]
        trename = data[optionanalysis]["tree_name"]
        df_data, df_mc = getdataframe_datamc(filedata, filemc, trename, mylistvariablesall)
        df_data, df_mc = filterdataframe_datamc(df_data, df_mc, var_skimming, varmin, varmax)
        df_ml, _, _ = preparemlsample(
            mltype, mlsubtype, optionanalysis, df_data, df_mc, nevents)
        df_ml = shuffle(df_ml)
test()
