###############################################################
##                                                           ##
##     Software for single-label classification with Scikit  ##
##      Origin: G.M.Innocenti (CERN)(ginnocen@cern.ch)       ##
##                                                           ##
###############################################################
from myimports import *
from utilitiesModels import getclassifiers,fit,test,savemodels,importanceplotall,decisionboundaries
from BinaryMultiFeaturesClassification import getvariablestraining,getvariablesothers,getvariableissignal,getvariablesall,getvariablecorrelation,getgridsearchparameters,getDataMCfiles,getTreeName,getdataframe,prepareMLsample
from utilitiesPerformance import precision_recall,plot_learning_curves,confusion,precision_recall,plot_learning_curves,cross_validation_mse,plot_cross_validation_mse
from utilitiesPCA import GetPCADataFrameAndPC,GetDataFrameStandardised,plotvariancePCA
from utilitiesCorrelations import scatterplot,correlationmatrix,vardistplot
from utilitiesGeneral import filterdataframe_pt,splitdataframe_sigbkg,checkdir
from utilitiesGridSearch import do_gridsearch,plot_gridsearch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

############### this is the only place where you should change parameters ################
optionClassification="Ds"
nevents=1000
ptmin=0
ptmax=100
suffix="Nevents%d_BinaryClassification%s_ptmin%d_ptmax%d" % (nevents,optionClassification,ptmin,ptmax)
var_pt="pt_cand_ML"
var_signal="signal_ML"

############### activate your channel ################
dosampleprep=0
docorrelation=0
doStandard=0
doPCA=0
dotraining=0
doimportance=0
dotesting=0
docrossvalidation=0
doRoCLearning=0
doBoundary=1
doBinarySearch=0
ncores=-1

##########################################################################################
# var_pt="pt_cand_ML"
# var_signal="signal_ML"
# path = "./plots/%.1f_%.1f_GeV"%(ptmin,ptmax)
# checkdir(path)

classifiers, names=getclassifiers()
mylistvariables=getvariablestraining(optionClassification)
mylistvariablesothers=getvariablesothers(optionClassification)
myvariablesy=getvariableissignal(optionClassification)
mylistvariablesx,mylistvariablesy=getvariablecorrelation(optionClassification)
mylistvariablesall=getvariablesall(optionClassification)


if(dosampleprep==1): 
  ### get the dataframes
  fileData,fileMC=getDataMCfiles(optionClassification)
  trename=getTreeName(optionClassification)
  dataframeData=getdataframe(fileData,trename,mylistvariablesall)
  dataframeMC=getdataframe(fileMC,trename,mylistvariablesall)
  ### select in pt
  dataframeData=filterdataframe_pt(dataframeData,var_pt,ptmin,ptmax)
  dataframeMC=filterdataframe_pt(dataframeMC,var_pt,ptmin,ptmax)
  ### prepare ML sample
  dataframeML=prepareMLsample(optionClassification,dataframeData,dataframeMC,nevents)
  dataframeML=shuffle(dataframeML)
  ### split in training/testing sample
  train_set, test_set = train_test_split(dataframeML, test_size=0.2, random_state=42)
  ### save the dataframes
  dataframeML.to_pickle("dataframes/dataframeML%s.pkl" % (suffix))
  dataframeML.to_csv("dataframes/dataframeML%s.csv" % (suffix))
  train_set.to_pickle("dataframes/trainsampleN%s.pkl" % (suffix))
  test_set.to_pickle("dataframes/testsampleN%s.pkl" % (suffix))

train_set = pd.read_pickle("dataframes/trainsampleN%s.pkl" % (suffix))
test_set = pd.read_pickle("dataframes/testsampleN%s.pkl" % (suffix))

# train_set = pd.read_pickle("../buildsampleCandBased/trainsampleSignalN10000BkgN10000PreMassCutDs.pkl")
# test_set = pd.read_pickle("../buildsampleCandBased/testsampleSignalN10000BkgN10000PreMassCutDs.pkl")
X_train= train_set[mylistvariables]
y_train=train_set[myvariablesy]
X_test= test_set[mylistvariables]
y_test=test_set[myvariablesy]

trainedmodels=[]

if(docorrelation==1):
  train_set_ptsel_sig,train_set_ptsel_bkg=splitdataframe_sigbkg(train_set,var_signal)
  vardistplot(train_set_ptsel_sig, train_set_ptsel_bkg,mylistvariables,"plots")
  scatterplot(train_set_ptsel_sig, train_set_ptsel_bkg,mylistvariablesx,mylistvariablesy,"plots")
  correlationmatrix(train_set_ptsel_sig,"plots","signal")
  correlationmatrix(train_set_ptsel_bkg,"plots","background")

if (doStandard==1):
  X_train=GetDataFrameStandardised(X_train)

if (doPCA==1):
  n_pca=9
  X_train,pca=GetPCADataFrameAndPC(X_train,n_pca)
  plotvariancePCA(pca,"plots")

if (dotraining==1):
  trainedmodels=fit(names, classifiers,X_train,y_train)
  savemodels(names,trainedmodels,"output",suffix)
  
if (doimportance==1):
  importanceplotall(mylistvariables,names,trainedmodels,suffix)
  
if (docrossvalidation==1): 
  df_scores=cross_validation_mse(names,classifiers,X_train,y_train,10,ncores)
  plot_cross_validation_mse(names,df_scores,suffix)

if (doRoCLearning==1):
#   confusion(mylistvariables,names,classifiers,suffix,X_train,y_train,5)
  precision_recall(mylistvariables,names,classifiers,suffix,X_train,y_train,5)
  plot_learning_curves(names,classifiers,suffix,X_train,y_train,100,3000,300)
  
if (dotesting==1):
  filenametest_set_ML="output/testsample%sMLdecision.pkl" % (suffix)
  ntuplename="fTreeFlagged%s" % (optionClassification)
  test_setML=test(names,trainedmodels,X_test,test_set)
  test_set.to_pickle(filenametest_set_ML)

if (doBoundary==1):
  mydecisionboundaries=decisionboundaries(names,trainedmodels,suffix,X_train,y_train)
  X_train_2PC,pca=GetPCADataFrameAndPC(X_train,2)
  trainedmodels=fit(names, classifiers,X_train_2PC,y_train)
  mydecisionboundaries=decisionboundaries(names,trainedmodels,suffix+"PCAdecomposition",X_train_2PC,y_train)

if (doBinarySearch==1):
  namesCV,classifiersCV,param_gridCV,changeparameter=getgridsearchparameters(optionClassification)
  grid_search_models,grid_search_bests=do_gridsearch(namesCV,classifiersCV,mylistvariables,param_gridCV,X_train,y_train,3,ncores)
  savemodels(names,grid_search_models,"output","GridSearchCV"+suffix)
  plot_gridsearch(namesCV,changeparameter,grid_search_models,"plots",suffix)


