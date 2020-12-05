# D2H HM weights

In this folder are placed the new weights for the HM MC, in which the data n_tracklet distribution is corrected for the trigger efficiency histogram.

##Files:

mcweights.root = weights estimated with a D in a D mass range. This is the file that has to be used for the analysis.

mcweights_withD.root = weights estimated with a D candidate. This is the file that has to be used for the weight sistematic estimate.

mcweight_SelEv = weights estimated with all the selected events. This file must not be used for the systematic.

mcweightsCristina = old weights estimated by Cristina. They are not corrected for the trigger efficiences. Use this file if the trigger correction for the HM is not enabled.





## WeightFromFunc Folder:
In the folder WeightFromFunc are placed the weights corrected for the trigger efficiency function (not histogram). They must not be used for the analysis.

mcweights.root = weights estimated with a D in a D mass range.

mcweight_SelEv = weights estimated with all the selected events.











