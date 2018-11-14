###############################################################
##                                                           ##
##     Software for single-label classification with Scikit  ##
##      Origin: G.M. Innocenti (CERN)(ginnocen@cern.ch)       ##
##                                                           ##
###############################################################

"""
Methods to: read and write ROOT TNtuple
"""

import array
import numpy as np
from ROOT import TNtuple

def ReadNTuple(ntuple, variables) : 
  """
    Return a numpy array with the values from TNtuple. 
      ntuple : input TNtuple
      variables : list of ntuple variables to read
  """
  data = []
  code_list = []
  for v in variables : 
    code_list += [ compile("i.%s" % v, '<string>', 'eval') ]
  nentries = ntuple.GetEntries()
  nvars = len(variables)
  array = np.zeros( (nentries, nvars) )
  for n,i in enumerate(ntuple) : 
    for m,v in enumerate(code_list) :
      array[n][m] = eval(v)
    if n % 100000 == 0 : 
      print (n, "/", nentries)
  return array

def ReadNTupleML(ntuple, variablesfeatures, variablesothers, variabley ) : 
  """
    Return a numpy array with the values from TNtuple. 
      ntuple : input TNtuple
      variables : list of ntuple variables to read
  """
  code_listfeatures = []
  code_listothers = []
  for v in variablesfeatures : 
    code_listfeatures += [ compile("i.%s" % v, '<string>', 'eval') ]
  for v in variablesothers : 
    code_listothers += [ compile("i.%s" % v, '<string>', 'eval') ]
  codevariabley = compile("i.%s" % variabley, '<string>', 'eval')
  nentries = ntuple.GetEntries()
  nvars = len(variablesfeatures)
  nvarsothers = len(variablesothers)
  arrayfeatures = np.zeros( (nentries, nvars) )
  arrayothers = np.zeros( (nentries, nvarsothers) )
  arrayy = np.zeros(nentries)
  for n,i in enumerate(ntuple) : 
    for m,v in enumerate(code_listfeatures) :
      arrayfeatures[n][m] = eval(v)
    for m,v in enumerate(code_listothers) :
      arrayothers[n][m] = eval(v)
    arrayy[n] = eval(codevariabley)
    if n % 100000 == 0 : 
      print (n, "/", nentries)
  return arrayfeatures,arrayothers, arrayy

def FillNTuple(tupname, data, names) : 
  """
    Create and fill ROOT NTuple with the data sample. 
      tupname : name of the NTuple
      data : data sample
      names : names of the NTuple variables
  """
  variables = ""
  for n in names : variables += "%s:" % n
  variables = variables[:-1]
  values = len(names)*[ 0. ]
  avalues = array.array('f', values)
  nt = TNtuple(tupname, "", variables)
  for d in data : 
    for i in range(len(names)) : avalues[i] = d[i]
    nt.Fill(avalues)
  nt.Write()
  