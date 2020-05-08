import pickle
import lz4.frame
df = pickle.load(lz4.frame.open("AnalysisResultsReco4_6_std.pkl.lz4", "rb"))

minpt = 5
maxpt = 10
minshape = 0.1
maxshape = 0.5

def function_010(shape, pt, shapemin, shapemax, ptmin, ptmax):

    value =  -1 *((-0.2 * ptmax + 0.2 * ptmin) * (-shape + shapemax) + \
              (- pt + ptmin) * (0.2 * shapemax - 0.2 * shapemin) + \
              -0.8 * (ptmax * (shapemax - shapemin) + ptmin * \
              (-shapemax + shapemin)))/(ptmax * (shapemax - shapemin) \
              + ptmin * (-shapemax + shapemin))
    return value

print(function_010(maxshape, minpt, minshape, maxshape, minpt, maxpt))
print(function_010(minshape, maxpt, minshape, maxshape, minpt, maxpt))
print(function_010(maxshape, maxpt, minshape, maxshape, minpt, maxpt))
print(function_010(minshape, minpt, minshape, maxshape, minpt, maxpt))
print(function_010(0.3, 7, minshape, maxshape, minpt, maxpt))
print(function_010(0.2, 8, minshape, maxshape, minpt, maxpt))

shape = df["zg_gen_jet"].values
pt = df["pt_gen_jet"].values

weight = [function_010(shape[i], pt[i], minshape, maxshape, minpt, maxpt) for i in range(len(shape))]
df["weights"] = weight

from ROOT import TH2F, TFile
import numpy as np
from root_numpy import fill_hist
from machine_learning_hep.utilities_plot import build2dhisto, fill2dhist
h = TH2F("h", "h", 5, 0.1, 0.5, 5, 5, 10)
hweight = TH2F("hweight", "h", 5, 0.1, 0.5, 5, 5, 10)

arr = df[["zg_gen_jet", "pt_gen_jet"]].to_numpy()
weights = df["weights"].to_numpy()
weightsp = [abs(i) for i in weights]
for index in range(len(arr)):
    h.Fill(shape[index], pt[index])
    hweight.Fill(shape[index], pt[index], weights[index])
f = TFile("file.root", "recreate")
f.cd()
h.Write()
hweight.Write()
