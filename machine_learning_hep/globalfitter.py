import os
import sys
import yaml
import math
#import numpy as np
from   array import array
from   ROOT  import TH1F, TF1, TFile
from   ROOT  import gROOT, gStyle
from   ROOT  import kBlue, kGray
from   ROOT  import TCanvas, TPaveText, Double

def fitter(histo, case, sgnfunc, bkgfunc, masspeak, ..., outputfolder):
    if "Lc" in case:
        print("add my fitter")
