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
from machine_learning_hep.general import get_database_ml_parameters

def preparemlsample(mltype, mlsubtype, case, dataframe_data, dataframe_mc, nevents):
    dataframe_ml_joined = pd.DataFrame()
    data = get_database_ml_parameters()
    if mltype == "BinaryClassification":

        if mlsubtype == "HFmeson":
            dataframe_bkg = dataframe_data
            dataframe_sig = dataframe_mc
            fmassmin, fmassmax = data[case]["mass_cut"]
            dataframe_sig = dataframe_sig.loc[
                (dataframe_sig["cand_type_ML"] == 2) | (dataframe_sig["cand_type_ML"] == 3)]
            dataframe_sig['signal_ML'] = 1
            dataframe_bkg = dataframe_bkg.loc[(dataframe_bkg["inv_mass_ML"] < fmassmin) | (
                dataframe_bkg["inv_mass_ML"] > fmassmax)]
            dataframe_bkg['signal_ML'] = 0

        if mlsubtype == "PID":
            dataframe_mc["pdg0_ML"] = dataframe_mc["pdg0_ML"].abs()
            dataframe_sig = dataframe_mc.loc[(dataframe_mc["pdg0_ML"] == data[case]["pdg_code"])]
            dataframe_sig['signal_ML'] = 1
            dataframe_bkg = dataframe_mc.loc[(dataframe_mc["pdg0_ML"] != data[case]["pdg_code"])]
            dataframe_bkg['signal_ML'] = 0

        if mlsubtype == "jettagging":
            dataframe_bkg = dataframe_mc
            dataframe_sig = dataframe_mc
            if case == "lightquarkjet":
                dataframe_sig = dataframe_sig.loc[
                    (dataframe_sig["Parton_Flag_ML"] == 1) |
                    (dataframe_sig["Parton_Flag_ML"] == 2) |
                    (dataframe_sig["Parton_Flag_ML"] == 3) |
                    (dataframe_sig["Parton_Flag_ML"] == 4) |
                    (dataframe_sig["Parton_Flag_ML"] == 5)]
                dataframe_bkg = dataframe_bkg.loc[(dataframe_bkg["Parton_Flag_ML"] > 5)]
            dataframe_sig['signal_ML'] = 1
            dataframe_bkg['signal_ML'] = 0

        if mlsubtype == "nuclei":
            dataframe_bkg = dataframe_mc
            dataframe_sig = dataframe_mc
            if case == "hypertritium":
                dataframe_sig = dataframe_sig.loc[(dataframe_sig["signal"] == 1)]
                dataframe_bkg = dataframe_bkg.loc[(dataframe_bkg["signal"] == -1)]
            dataframe_sig['signal_ML'] = 1
            dataframe_bkg['signal_ML'] = 0

    dataframe_sig = dataframe_sig[:nevents]
    dataframe_bkg = dataframe_bkg[:nevents]
    dataframe_ml_joined = pd.concat([dataframe_sig, dataframe_bkg])
    if ((nevents > len(dataframe_sig)) or (nevents > len(dataframe_bkg))):
        print("------- ERROR: there are not so many events!!!!!! -------")

    return dataframe_ml_joined, dataframe_sig, dataframe_bkg
